"""
ComfyUI-TopazVideoAI 视频加载节点。

从 ComfyUI-VideoHelperSuite (VHS) 的 LoadVideoFFmpeg 系列移植而来，
目的是避免 VHS 更新时丢失对 max_duration (时间上限) 的修改。

与 VHS 原版的区别:
  - 读取上限用 max_duration (秒) 而非 frame_load_cap (帧)，与 start_time 单位一致，方便拼接
  - 复用 Topaz 插件自带的 ffmpeg 探测逻辑 (detect_topaz_install_dir)
  - 去掉了 meta_batch / vae / format 帧对齐等 Topaz 场景不需要的复杂逻辑
  - 节点类名改为 Topaz* 前缀，避免与 VHS 冲突
  - 返回类型保持 AUDIO / VHS_VIDEOINFO，可与 VHS 生态节点互通

节点列表:
  TopazLoadVideoFFmpeg     - 从 ComfyUI input 目录选择视频 (下拉)
  TopazLoadVideoFFmpegPath - 从任意路径加载视频 (字符串)
两个节点都输出 IMAGE / frame_count / mask / audio / video_info。
"""
import os
import re
import time
import subprocess
import itertools
import logging
from collections.abc import Mapping

import numpy as np
import torch

import folder_paths
from comfy.utils import ProgressBar

# 复用主模块的 Topaz ffmpeg 探测 + 环境变量逻辑
from .topaz_video_node import detect_topaz_install_dir, topaz_env_for_subprocess

logger = logging.getLogger('TopazVideoAI')

# 常量
BIGMAX = (2 ** 53 - 1)
DIMMAX = 8192
ENCODE_ARGS = ("utf-8", 'backslashreplace')
video_extensions = ['webm', 'mp4', 'mkv', 'gif', 'mov', 'avi', 'flv', 'ts', 'm4v', 'mpg', 'mpeg']


# ---------------------------------------------------------------------------
# 辅助函数 (从 VHS utils.py 内联，去掉不相关的逻辑)
# ---------------------------------------------------------------------------

def _get_ffmpeg():
    """获取 ffmpeg 可执行文件路径。优先 Topaz 自带的，其次系统的。"""
    topaz_dir = detect_topaz_install_dir()
    exe = os.path.join(topaz_dir, 'ffmpeg.exe' if os.name == 'nt' else 'ffmpeg')
    if os.path.isfile(exe):
        return exe
    return 'ffmpeg'  # 回退到 PATH


def _strip_path(path):
    path = path.strip()
    if path.startswith('"'):
        path = path[1:]
    if path.endswith('"'):
        path = path[:-1]
    return path


def _is_url(url):
    return url.split("://")[0] in ["http", "https"]


def _calculate_file_hash(filename, hash_every_n=1):
    """计算文件哈希，用于 IS_CHANGED 缓存失效。"""
    import hashlib
    try:
        with open(filename, "rb") as f:
            m = hashlib.sha256()
            size = os.path.getsize(filename)
            hash_every_n = max(1, int(size / (1024 * 1024))) if size > 1024 * 1024 else 1
            i = 0
            for chunk in iter(lambda: f.read(4096 * hash_every_n), b""):
                m.update(chunk)
                i += 1
            return m.hexdigest() + str(size) + str(i)
    except (OSError, IOError):
        return ""


def _hash_path(path):
    if path is None:
        return "input"
    if _is_url(path):
        return "url"
    if not os.path.isfile(_strip_path(path)):
        return "DNE"
    return _calculate_file_hash(_strip_path(path))


def _target_size(width, height, custom_width, custom_height, downscale_ratio=8):
    """计算目标尺寸，对齐到 downscale_ratio 的倍数。"""
    if downscale_ratio is None:
        downscale_ratio = 8
    if custom_width == 0 and custom_height == 0:
        pass
    elif custom_height == 0:
        height *= custom_width / width
        width = custom_width
    elif custom_width == 0:
        width *= custom_height / height
        height = custom_height
    else:
        width = custom_width
        height = custom_height
    width = int(width / downscale_ratio + 0.5) * downscale_ratio
    height = int(height / downscale_ratio + 0.5) * downscale_ratio
    return (width, height)


# ---------------------------------------------------------------------------
# 音频提取 (从 VHS utils.py 内联)
# ---------------------------------------------------------------------------

def _get_audio(ffmpeg_path, file, start_time=0, duration=0):
    args = [ffmpeg_path, "-i", file]
    if start_time > 0:
        args += ["-ss", str(start_time)]
    if duration > 0:
        args += ["-t", str(duration)]
    try:
        res = subprocess.run(args + ["-f", "f32le", "-"],
                             capture_output=True, check=True,
                             env=topaz_env_for_subprocess())
        audio = torch.frombuffer(bytearray(res.stdout), dtype=torch.float32)
        match = re.search(r', (\d+) Hz, (\w+), ', res.stderr.decode(*ENCODE_ARGS))
    except subprocess.CalledProcessError as e:
        raise Exception(f"Failed to extract audio from {file}:\n"
                        + e.stderr.decode(*ENCODE_ARGS))
    if match:
        ar = int(match.group(1))
        ac = {"mono": 1, "stereo": 2}.get(match.group(2), 2)
    else:
        ar = 44100
        ac = 2
    audio = audio.reshape((-1, ac)).transpose(0, 1).unsqueeze(0)
    return {'waveform': audio, 'sample_rate': ar}


class _LazyAudioMap(Mapping):
    """延迟加载音频，只有真正访问时才调用 ffmpeg 提取。"""
    def __init__(self, ffmpeg_path, file, start_time, duration):
        self.ffmpeg_path = ffmpeg_path
        self.file = file
        self.start_time = start_time
        self.duration = duration
        self._dict = None

    def _ensure(self):
        if self._dict is None:
            self._dict = _get_audio(self.ffmpeg_path, self.file, self.start_time, self.duration)

    def __getitem__(self, key):
        self._ensure()
        return self._dict[key]

    def __iter__(self):
        self._ensure()
        return iter(self._dict)

    def __len__(self):
        self._ensure()
        return len(self._dict)


def _lazy_get_audio(ffmpeg_path, file, start_time=0, duration=0):
    return _LazyAudioMap(ffmpeg_path, file, start_time, duration)


# ---------------------------------------------------------------------------
# ffmpeg 帧生成器 (核心: 从 VHS 移植，frame_load_cap 改为 max_duration)
# ---------------------------------------------------------------------------

def _ffmpeg_frame_generator(ffmpeg_path, video, force_rate, max_duration, start_time,
                            custom_width, custom_height, downscale_ratio=8):
    """
    用 ffmpeg 读取视频，逐帧 yield numpy 数组 (H,W,C) float32 [0,1]。
    先 yield 一个元组 (视频元信息)，再逐帧 yield。
    max_duration 和 start_time 都是秒。
    """
    args_input = ["-i", video]
    args_dummy = [ffmpeg_path] + args_input + ['-c', 'copy', '-frames:v', '1', "-f", "null", "-"]
    size_base = None
    fps_base = None
    try:
        dummy_res = subprocess.run(args_dummy, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.PIPE, check=True,
                                   env=topaz_env_for_subprocess())
    except subprocess.CalledProcessError as e:
        raise Exception("An error occurred in the ffmpeg subprocess:\n"
                        + e.stderr.decode(*ENCODE_ARGS))
    lines = dummy_res.stderr.decode(*ENCODE_ARGS)
    if "Video: vp9 " in lines:
        args_input = ["-c:v", "libvpx-vp9"] + args_input
        args_dummy = [ffmpeg_path] + args_input + ['-c', 'copy', '-frames:v', '1', "-f", "null", "-"]
        try:
            dummy_res = subprocess.run(args_dummy, stdout=subprocess.DEVNULL,
                                       stderr=subprocess.PIPE, check=True,
                                       env=topaz_env_for_subprocess())
        except subprocess.CalledProcessError as e:
            raise Exception("An error occurred in the ffmpeg subprocess:\n"
                            + e.stderr.decode(*ENCODE_ARGS))
        lines = dummy_res.stderr.decode(*ENCODE_ARGS)

    for line in lines.split('\n'):
        match = re.search(r"^ *Stream .* Video.*, ([1-9]|\d{2,})x(\d+)", line)
        if match is not None:
            size_base = [int(match.group(1)), int(match.group(2))]
            fps_match = re.search(r", ([\d.]+) fps", line)
            if fps_match:
                fps_base = float(fps_match.group(1))
            else:
                fps_base = 1
            alpha = re.search(r"(yuva|rgba|bgra|gbra)", line) is not None
            break
    else:
        raise Exception("Failed to parse video/image information. FFMPEG output:\n" + lines)

    durs_match = re.search(r"Duration: (\d+:\d+:\d+\.\d+),", lines)
    if durs_match:
        durs = durs_match.group(1).split(':')
        duration = int(durs[0]) * 3600 + int(durs[1]) * 60 + float(durs[2])
    else:
        duration = 0

    if start_time > 0:
        if start_time > 4:
            post_seek = ['-ss', '4']
            args_input = ['-ss', str(start_time - 4)] + args_input
        else:
            post_seek = ['-ss', str(start_time)]
    else:
        post_seek = []
    args_all_frames = [ffmpeg_path, "-v", "error", "-an"] + \
        args_input + ["-pix_fmt", "rgba64le"] + post_seek

    vfilters = []
    if force_rate != 0:
        vfilters.append("fps=fps=" + str(force_rate))
    if custom_width != 0 or custom_height != 0:
        size = _target_size(size_base[0], size_base[1], custom_width,
                            custom_height, downscale_ratio=downscale_ratio)
        ar = float(size[0]) / size[1]
        if abs(size_base[0] * ar - size_base[1]) >= 1:
            vfilters.append(f"crop=if(gt({ar}\\,a)\\,iw\\,ih*{ar}):if(gt({ar}\\,a)\\,iw/{ar}\\,ih)")
        size_arg = ':'.join(map(str, size))
        vfilters.append(f"scale={size_arg}")
    else:
        size = size_base
    if len(vfilters) > 0:
        args_all_frames += ["-vf", ",".join(vfilters)]
    yieldable_frames = (force_rate or fps_base) * duration
    if max_duration > 0:
        args_all_frames += ["-t", str(max_duration)]
        yieldable_frames = min(yieldable_frames, max_duration * (force_rate or fps_base))
    yield (size_base[0], size_base[1], fps_base, duration, fps_base * duration,
          1 / (force_rate or fps_base), yieldable_frames, size[0], size[1], alpha)

    args_all_frames += ["-f", "rawvideo", "-"]
    pbar = ProgressBar(int(yieldable_frames))
    try:
        with subprocess.Popen(args_all_frames, stdout=subprocess.PIPE,
                              env=topaz_env_for_subprocess()) as proc:
            bpi = size[0] * size[1] * 8  # rgba64 = 8 bytes/pixel
            current_bytes = bytearray(bpi)
            current_offset = 0
            prev_frame = None
            while True:
                bytes_read = proc.stdout.read(bpi - current_offset)
                if bytes_read is None:
                    time.sleep(.1)
                    continue
                if len(bytes_read) == 0:
                    break
                current_bytes[current_offset:len(bytes_read)] = bytes_read
                current_offset += len(bytes_read)
                if current_offset == bpi:
                    if prev_frame is not None:
                        yield prev_frame
                        pbar.update(1)
                    prev_frame = np.frombuffer(current_bytes, dtype=np.dtype(np.uint16).newbyteorder("<")).reshape(size[1], size[0], 4) / (2**16 - 1)
                    if not alpha:
                        prev_frame = prev_frame[:, :, :-1]
                    current_offset = 0
    except BrokenPipeError:
        raise Exception("An error occured in the ffmpeg subprocess:\n"
                        + proc.stderr.read().decode(*ENCODE_ARGS))
    if prev_frame is not None:
        yield prev_frame


# ---------------------------------------------------------------------------
# 核心加载逻辑 (从 VHS load_video 精简，去掉 meta_batch/vae/format)
# ---------------------------------------------------------------------------

def _do_load_video(ffmpeg_path, video, force_rate, max_duration, start_time,
                   custom_width, custom_height):
    """
    加载视频，返回 (images, frame_count, mask, audio, video_info)。
    images: torch.Tensor (N,H,W,3) float32
    mask: torch.Tensor (N,H,W) float32 (alpha 通道反转，无 alpha 时为 0)
    audio: 延迟加载的 AUDIO dict {'waveform':..., 'sample_rate':...}
    video_info: dict (兼容 VHS_VIDEOINFO)
    """
    video = _strip_path(video)
    downscale_ratio = 1  # 不做 VAE 对齐
    gen = _ffmpeg_frame_generator(ffmpeg_path, video, force_rate, max_duration, start_time,
                                 custom_width, custom_height, downscale_ratio)
    (width, height, fps, duration, total_frames, target_frame_time, yieldable_frames,
     new_width, new_height, alpha) = next(gen)

    # 内存安全: 限制最大帧数 (预留 ~128MB)
    try:
        import psutil
        memory_limit = (psutil.virtual_memory().available + psutil.swap_memory().free) - 2 ** 27
    except Exception:
        memory_limit = BIGMAX
    max_loadable_frames = int(memory_limit // (width * height * 3 * 0.1))
    gen = itertools.islice(gen, max_loadable_frames)

    # 收集帧 (fromiter 比 append 高效)
    images = torch.from_numpy(np.fromiter(gen, np.dtype((np.float32, (new_height, new_width, 4 if alpha else 3)))))
    if len(images) == 0:
        raise RuntimeError("No frames generated")

    frame_count = len(images)
    if alpha:
        # 拆分 RGB 和 alpha (alpha 作为 mask，反转: 1=透明区)
        mask = 1 - images[:, :, :, 3]
        images = images[:, :, :, :3]
    else:
        mask = torch.zeros(frame_count, 64, 64)

    # 音频延迟加载 (start_time 和 max_duration 都是秒)
    cap_duration = max_duration if max_duration > 0 else 0
    audio = _lazy_get_audio(ffmpeg_path, video, start_time, cap_duration)

    video_info = {
        "source_fps": fps,
        "source_frame_count": total_frames,
        "source_duration": duration,
        "source_width": width,
        "source_height": height,
        "loaded_fps": 1 / target_frame_time,
        "loaded_frame_count": frame_count,
        "loaded_duration": frame_count * target_frame_time,
        "loaded_width": new_width,
        "loaded_height": new_height,
    }
    return (images, frame_count, mask, audio, video_info)


# ---------------------------------------------------------------------------
# 节点类
# ---------------------------------------------------------------------------

class TopazLoadVideoFFmpeg:
    """从 ComfyUI input 目录选择视频加载 (下拉列表)。"""
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = []
        for f in os.listdir(input_dir):
            if os.path.isfile(os.path.join(input_dir, f)):
                file_parts = f.split('.')
                if len(file_parts) > 1 and (file_parts[-1].lower() in video_extensions):
                    files.append(f)
        return {"required": {
                    "video": (sorted(files),),
                    "force_rate": ("FLOAT", {"default": 0, "min": 0, "max": 60, "step": 1, "disable": 0}),
                    "custom_width": ("INT", {"default": 0, "min": 0, "max": DIMMAX, 'disable': 0}),
                    "custom_height": ("INT", {"default": 0, "min": 0, "max": DIMMAX, 'disable': 0}),
                    "max_duration": ("FLOAT", {"default": 0, "min": 0, "max": BIGMAX, "step": 0.001, "disable": 0}),
                    "start_time": ("FLOAT", {"default": 0, "min": 0, "max": BIGMAX, "step": 0.001}),
                },
                "optional": {
                    "meta_batch": ("VHS_BatchManager",),
                },
                "hidden": {
                    "unique_id": "UNIQUE_ID"
                },
        }

    CATEGORY = "TopazVideoAI"
    RETURN_TYPES = ("IMAGE", "INT", "MASK", "AUDIO", "VHS_VIDEOINFO")
    RETURN_NAMES = ("IMAGE", "frame_count", "mask", "audio", "video_info")
    FUNCTION = "load_video"

    def load_video(self, **kwargs):
        kwargs['video'] = folder_paths.get_annotated_filepath(_strip_path(kwargs['video']))
        kwargs.pop('meta_batch', None)
        kwargs.pop('unique_id', None)
        ffmpeg_path = _get_ffmpeg()
        return _do_load_video(ffmpeg_path, **kwargs)

    @classmethod
    def IS_CHANGED(cls, video, **kwargs):
        image_path = folder_paths.get_annotated_filepath(video)
        return _calculate_file_hash(image_path)

    @classmethod
    def VALIDATE_INPUTS(cls, video):
        if not folder_paths.exists_annotated_filepath(video):
            return "Invalid video file: {}".format(video)
        return True


class TopazLoadVideoFFmpegPath:
    """从任意文件系统路径加载视频。"""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                    "video": ("STRING", {"placeholder": "X://insert/path/here.mp4"}),
                    "force_rate": ("FLOAT", {"default": 0, "min": 0, "max": 60, "step": 1, "disable": 0}),
                    "custom_width": ("INT", {"default": 0, "min": 0, "max": DIMMAX, 'disable': 0}),
                    "custom_height": ("INT", {"default": 0, "min": 0, "max": DIMMAX, 'disable': 0}),
                    "max_duration": ("FLOAT", {"default": 0, "min": 0, "max": BIGMAX, "step": 0.001, "disable": 0}),
                    "start_time": ("FLOAT", {"default": 0, "min": 0, "max": BIGMAX, "step": 0.001}),
                },
                "optional": {
                    "meta_batch": ("VHS_BatchManager",),
                },
                "hidden": {
                    "unique_id": "UNIQUE_ID"
                },
        }

    CATEGORY = "TopazVideoAI"
    RETURN_TYPES = ("IMAGE", "INT", "MASK", "AUDIO", "VHS_VIDEOINFO")
    RETURN_NAMES = ("IMAGE", "frame_count", "mask", "audio", "video_info")
    FUNCTION = "load_video"

    def load_video(self, **kwargs):
        video = _strip_path(kwargs['video'])
        if video is None or not os.path.isfile(video):
            raise Exception("video is not a valid path: " + str(video))
        kwargs.pop('meta_batch', None)
        kwargs.pop('unique_id', None)
        ffmpeg_path = _get_ffmpeg()
        return _do_load_video(ffmpeg_path, **kwargs)

    @classmethod
    def IS_CHANGED(cls, video, **kwargs):
        return _hash_path(video)

    @classmethod
    def VALIDATE_INPUTS(cls, video):
        if not os.path.isfile(_strip_path(video)):
            return "Invalid video file: {}".format(video)
        return True
