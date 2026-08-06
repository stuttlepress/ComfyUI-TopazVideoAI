import os
import json
import glob
import time
import numpy as np
import torch
import subprocess
import uuid
from PIL import Image
import tempfile
import logging
import shutil
from concurrent.futures import ThreadPoolExecutor
import re
import folder_paths

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('TopazVideoAI')


# ---------------------------------------------------------------------------
# Topaz 安装路径 / 模型目录自动探测
#
# Topaz 在 2025 年把产品从 "Topaz Video AI" 改名为 "Topaz Video":
#   - 安装目录: ...\Topaz Video AI  ->  ...\Topaz Video
#   - 模型目录: ...\Topaz Video AI\models  ->  ...\Topaz Video\models
# 旧版本用户的 TVAI_MODEL_DIR 环境变量还指向旧路径，这里同时兼容两种命名。
# ---------------------------------------------------------------------------
_TOPAZ_BASE = r"C:\Program Files\Topaz Labs LLC"
_PROGRAMDATA_BASE = r"C:\ProgramData\Topaz Labs LLC"

# 按优先级排列，新版 (Topaz Video) 在前
_TOPAZ_INSTALL_CANDIDATES = [
    os.path.join(_TOPAZ_BASE, "Topaz Video"),
    os.path.join(_TOPAZ_BASE, "Topaz Video AI"),
]

_TOPAZ_MODELDIR_CANDIDATES = [
    os.path.join(_PROGRAMDATA_BASE, "Topaz Video", "models"),
    os.path.join(_PROGRAMDATA_BASE, "Topaz Video AI", "models"),
]


def detect_topaz_install_dir():
    """返回第一个存在 ffmpeg.exe 的 Topaz 安装目录，找不到返回第一个候选。"""
    for cand in _TOPAZ_INSTALL_CANDIDATES:
        if os.path.isfile(os.path.join(cand, "ffmpeg.exe")):
            return cand
    # 兜底: 仍返回新版路径，让错误信息更明确
    return _TOPAZ_INSTALL_CANDIDATES[0]


def _is_real_model_dir(path):
    """
    判断一个目录是否是真正含 Topaz 模型的目录。
    旧版 'Topaz Video AI\\models' 在新版被改造成了 Python 运行环境
    (顶层只有 proxy.json + DLLs/Lib/include 等子目录)，需要排除。
    用 '是否含 modelType 字段的 json' 作为判据最可靠。
    """
    if not path or not os.path.isdir(path):
        return False
    # 快速判据：必须同时存在 .json 和模型数据文件 (.tz3/.tz)
    has_json = bool(glob.glob(os.path.join(path, "*.json")))
    has_data = bool(glob.glob(os.path.join(path, "*.tz3")) or
                    glob.glob(os.path.join(path, "*.tz")))
    if not (has_json and has_data):
        return False
    # 严格判据：至少有一个 json 带 modelType 字段 (排除 proxy.json 这类杂项)
    for f in glob.glob(os.path.join(path, "*.json")):
        try:
            with open(f, "r", encoding="utf-8") as fh:
                d = json.load(fh)
            if isinstance(d, dict) and "modelType" in d:
                return True
        except Exception:
            continue
    return False


def detect_topaz_model_dir():
    """
    返回真正含 Topaz 模型的目录。
    优先级:
      1. 自动探测的新版路径 'C:\\ProgramData\\Topaz Labs LLC\\Topaz Video\\models'
         (新版默认位置，且能避开旧版 TVAI_MODEL_DIR 环境变量残留)
      2. 自动探测的旧版路径 '...\\Topaz Video AI\\models'
      3. 环境变量 TVAI_MODEL_DIR / TVAI_MODEL_DATA_DIR (仅当通过真实性检查)
      4. 兜底返回第一个候选
    环境变量优先级降低，是因为大量从旧版升级的用户环境变量仍指向
    'Topaz Video AI\\models'，而该目录在新版已变成 Python 运行环境，
    会导致 'Model not found' 错误。
    """
    # 1-2. 自动探测候选路径 (新版优先)
    for cand in _TOPAZ_MODELDIR_CANDIDATES:
        if _is_real_model_dir(cand):
            return cand

    # 3. 环境变量 (需通过真实性检查)
    for c in [os.environ.get("TVAI_MODEL_DIR"), os.environ.get("TVAI_MODEL_DATA_DIR")]:
        if _is_real_model_dir(c):
            return c

    # 4. 兜底
    return _TOPAZ_MODELDIR_CANDIDATES[0]


def topaz_env_for_subprocess(model_dir=None):
    """
    构造调用 Topaz ffmpeg 时使用的环境变量字典。
    把可能失效的旧 TVAI_MODEL_DIR 覆盖为探测到的新模型目录，
    防止旧版残留环境变量导致的 'Model not found'。
    """
    env = os.environ.copy()
    md = model_dir or detect_topaz_model_dir()
    env["TVAI_MODEL_DIR"] = md
    env["TVAI_MODEL_DATA_DIR"] = md
    return env


# ---------------------------------------------------------------------------
# 模型扫描
#
# 新版模型 json 通过 modelType 字段区分用途:
#   modelType=1 -> 放大/修复类 (tvai_up)
#   modelType=2 -> 补帧类    (tvai_fi)
# 其它 modelType (3=auto param estimation, 4=shot detection,
# 5=ref, 8=slm 等) 不通过 tvai_up/tvai_fi 直接调用，故排除。
# astra* 系列 (modelType=None) 走独立的 enhancement 路径，tvai_up 无法调用，
# 同样排除以保证列表里每个模型都能稳定运行。
# ---------------------------------------------------------------------------
_MODEL_TYPE_UPSCALE = 1
_MODEL_TYPE_FI = 2


def _scan_models_by_type(model_type, model_dir=None):
    """扫描指定 modelType 的启用模型，返回排序后的 (shortname, displayName) 列表。"""
    md = model_dir or detect_topaz_model_dir()
    results = []
    if not os.path.isdir(md):
        return results
    for f in glob.glob(os.path.join(md, "*.json")):
        try:
            with open(f, "r", encoding="utf-8") as fh:
                d = json.load(fh)
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        if d.get("enabled", 1) != 1:
            continue
        if d.get("modelType") != model_type:
            continue
        shortname = os.path.splitext(os.path.basename(f))[0]
        display = d.get("displayName") or shortname
        results.append((shortname, display))
    results.sort(key=lambda x: x[0])
    return results


# 扫描结果在模块加载时计算一次并缓存。模型目录变更极少，无需频繁重扫。
# 提供 *_scan 函数可在节点里强制刷新（例如用户改了路径后重启 ComfyUI）。
_UPSCALE_MODELS_CACHE = None
_FI_MODELS_CACHE = None


def get_upscale_models():
    global _UPSCALE_MODELS_CACHE
    if _UPSCALE_MODELS_CACHE is None:
        _UPSCALE_MODELS_CACHE = _scan_models_by_type(_MODEL_TYPE_UPSCALE)
    return _UPSCALE_MODELS_CACHE


def get_fi_models():
    global _FI_MODELS_CACHE
    if _FI_MODELS_CACHE is None:
        _FI_MODELS_CACHE = _scan_models_by_type(_MODEL_TYPE_FI)
    return _FI_MODELS_CACHE


def _format_model_list(models):
    """把 (shortname, display) 列表转成 comfyui 用的短名列表。"""
    return [s for s, _ in models]


def _model_shortnames(models):
    return [s for s, _ in models]


# ---------------------------------------------------------------------------
# tvai_up 滤镜参数定义 (新版 ffmpeg 8.1)
#
# 旧版的 blend 参数已移除，拆分为更精细的参数。这里把每个参数定义成
# (filter_key, comfyui_field) 元组，便于节点 INPUT_TYPES 与滤镜链构建共享。
# 范围与默认值与 tvai_up 滤镜帮助一致 (-1 ~ 1，部分为 0~1)。
# ---------------------------------------------------------------------------
UPSCALE_FILTER_PARAMS = [
    # (filter_key, comfy_field, default, min, max, step)
    ("compression", "compression", 1.0, -1.0, 1.0, 0.1),
    ("noise",       "noise",       0.0, -1.0, 1.0, 0.1),
    ("details",     "details",     0.0, -1.0, 1.0, 0.1),
    ("halo",        "halo",        0.0, -1.0, 1.0, 0.1),
    ("blur",        "blur",        0.0, -1.0, 1.0, 0.1),
    ("preblur",     "preblur",     0.0, -1.0, 1.0, 0.1),
]

class TopazUpscaleParamsNode:
    """
    构造一组放大参数，可串接成滤镜链 (通过 previous_upscale 输入)。

    适配新版 Topaz Video (ffmpeg 8.1): 旧的单一 blend 参数已移除，
    拆分为 compression / noise / details / halo / blur / preblur 六个独立参数，
    分别对应 tvai_up 滤镜的同名选项，含义见滤镜帮助:
      - compression: 去除压缩块效应/蚊噪 (推荐默认 1.0)
      - noise:      去除 ISO 噪点 (负值偏重保细节)
      - details:    恢复相机降噪丢失的细节纹理
      - halo:       抑制过锐产生的振铃/光晕
      - blur:       额外锐化 (输入偏软时调高)
      - preblur:    预处理模糊 (抗锯齿/摩尔纹，负值偏重抗锯齿)
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        upscale_models = _model_shortnames(get_upscale_models()) or ["prob-4"]
        # 默认放大模型用 Proteus (通用性最强); 取已安装的最新 prob 版本
        default_model = "prob-4" if "prob-4" in upscale_models else (
            "prob-3" if "prob-3" in upscale_models else upscale_models[0]
        )
        required = {
            "upscale_factor": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.5}),
            "upscale_model": (upscale_models, {"default": default_model}),
        }
        # 新版细分参数 (每个都直接对应 tvai_up 的同名选项)
        for fkey, field, default, vmin, vmax, step in UPSCALE_FILTER_PARAMS:
            required[field] = ("FLOAT", {"default": default, "min": vmin, "max": vmax, "step": step})
        return {
            "required": required,
            "optional": {
                "previous_upscale": ("UPSCALE_PARAMS",),
            }
        }

    RETURN_TYPES = ("UPSCALE_PARAMS",)
    FUNCTION = "get_params"
    CATEGORY = "video"

    def get_params(self, upscale_factor=2.0, upscale_model="prob-4",
                   compression=1.0, noise=0.0, details=0.0, halo=0.0,
                   blur=0.0, preblur=0.0, previous_upscale=None):
        # thm-2 只支持 1x，强制修正避免滤镜初始化失败
        if upscale_model == "thm-2" and upscale_factor != 1.0:
            upscale_factor = 1.0
            logger.warning("thm-2 forces upscale_factor=1.0")

        current_params = {
            "upscale_factor": upscale_factor,
            "upscale_model": upscale_model,
            "compression": compression,
            "noise": noise,
            "details": details,
            "halo": halo,
            "blur": blur,
            "preblur": preblur,
        }

        if previous_upscale is None:
            return ([current_params],)
        else:
            return (previous_upscale + [current_params],)

class TopazVideoAINode:
    def __init__(self):
        self.base_temp_dir = tempfile.gettempdir()
        self.output_dir = os.path.join(self.base_temp_dir, "comfyui_topaz_temp")
        os.makedirs(self.output_dir, exist_ok=True)
        self.temp_files = []
        logger.debug(f"Initialized temp directory at: {self.output_dir}")
        if not CUPY_AVAILABLE:
            logger.warning("CuPy not available. Some GPU operations will be disabled.")

    @classmethod
    def INPUT_TYPES(cls):
        upscale_models = _model_shortnames(get_upscale_models()) or ["prob-4"]
        fi_models = _model_shortnames(get_fi_models()) or ["apo-8", "apf-1", "chr-2", "chf-3"]
        # 默认放大模型用 Proteus (prob-4)，它是通用性最强的修复/放大模型，适配各类素材；
        # 装了多个版本时取最新的 prob。
        default_upscale = "prob-4" if "prob-4" in upscale_models else (
            upscale_models[0] if not upscale_models else "prob-3" if "prob-3" in upscale_models else upscale_models[0]
        )
        default_fi = "apo-8" if "apo-8" in fi_models else fi_models[0]
        # 自动探测 Topaz 安装目录作为默认值 (新版 "Topaz Video" 优先，兼容旧版)
        default_topaz_path = detect_topaz_install_dir()

        required = {
            "images": ("IMAGE",),
            "enable_upscale": ("BOOLEAN", {"default": False}),
            "upscale_factor": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.5}),
            "upscale_model": (upscale_models, {"default": default_upscale}),
        }
        # 新版细分参数 (tvai_up 滤镜帮助里的同名选项)
        for fkey, field, default, vmin, vmax, step in UPSCALE_FILTER_PARAMS:
            required[field] = ("FLOAT", {"default": default, "min": vmin, "max": vmax, "step": step})

        required.update({
            "enable_interpolation": ("BOOLEAN", {"default": False}),
            "input_fps": ("INT", {"default": 24, "min": 1, "max": 240}),
            "interpolation_multiplier": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 8.0, "step": 0.5}),
            "interpolation_model": (fi_models, {"default": default_fi}),
            "use_gpu": ("BOOLEAN", {"default": True}),
            "topaz_ffmpeg_path": ("STRING", {"default": default_topaz_path}),
            "force_topaz_ffmpeg": ("BOOLEAN", {"default": True}),
            "save_video": ("BOOLEAN", {"default": False}),
            "filename_prefix": ("STRING", {"default": "TopazVideo"}),
        })
        return {
            "required": required,
            "optional": {
                "previous_upscale": ("UPSCALE_PARAMS",),
                # 可选音频输入，兼容 VHS 的 AUDIO 类型 ({'waveform': Tensor, 'sample_rate': int})。
                # 接了音频时，Topaz 处理完视频后会用第二个 ffmpeg 把音轨 mux 进去
                # (视频流 -c:v copy 不重编码，很快)。不接则输出纯视频。
                "audio": ("AUDIO",),
            },
            # hidden 输入让节点在执行时拿到完整 prompt 图谱和自身 id，
            # 用于判断 IMAGE 输出端口是否真的被下游连接 (见 _is_image_output_connected)。
            "hidden": {
                "prompt": "PROMPT",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    OUTPUT_NODE = True
    FUNCTION = "process_video"
    CATEGORY = "video"

    # IMAGE 在 RETURN_TYPES 中的位置 (索引 0)。下游连线以 [from_node_id, output_index]
    # 形式存储在 prompt 里，这里用来判断本节点的 IMAGE 端口是否被消费。
    _IMAGE_OUTPUT_INDEX = 0

    def _is_image_output_connected(self, prompt, unique_id):
        """
        扫描 prompt 图谱，判断本节点的 IMAGE 输出 (索引 0) 是否被任何下游节点连接。
        ComfyUI 把连线信息只存在"输入侧": 每个节点的 inputs 值若是 [from_id, from_slot]
        二元列表，就表示该输入来自 from_id 节点的第 from_slot 个输出。
        本节点只关心是否有别的节点把本节点 id + IMAGE 索引当作输入。
        参考 ComfyUI-VideoHelperSuite 的同类做法。
        """
        if not prompt or unique_id is None:
            # 无法判断时保守返回 True (即解码)，保证下游不断图
            return True
        uid = str(unique_id)
        target_slot = self._IMAGE_OUTPUT_INDEX
        for other_id, other_node in prompt.items():
            if str(other_id) == uid:
                continue
            if not isinstance(other_node, dict):
                continue
            for inp in other_node.get("inputs", {}).values():
                # 连线值是 [from_node_id, from_socket_index]
                if isinstance(inp, list) and len(inp) == 2 \
                   and str(inp[0]) == uid and inp[1] == target_slot:
                    return True
        return False

    def _get_system_ffmpeg(self):
        """Try to find system FFmpeg installation"""
        try:
            system_paths = os.environ.get("PATH", "").split(os.pathsep)
            for path in system_paths:
                topaz_ffmpeg_path = os.path.join(path, "ffmpeg.exe" if os.name == "nt" else "ffmpeg")
                if os.path.isfile(topaz_ffmpeg_path) and os.access(topaz_ffmpeg_path, os.X_OK):
                    logger.debug(f"Found system FFmpeg at: {topaz_ffmpeg_path}")
                    return topaz_ffmpeg_path

            if os.name == "nt":
                common_locations = [
                    os.path.expandvars(r"%ProgramFiles%\FFmpeg\bin\ffmpeg.exe"),
                    os.path.expandvars(r"%ProgramFiles(x86)%\FFmpeg\bin\ffmpeg.exe"),
                    os.path.expandvars(r"%USERPROFILE%\FFmpeg\bin\ffmpeg.exe")
                ]
                for location in common_locations:
                    if os.path.isfile(location) and os.access(location, os.X_OK):
                        logger.debug(f"Found system FFmpeg at: {location}")
                        return location

            result = subprocess.run(['ffmpeg', '-version'], 
                                capture_output=True, 
                                text=True, 
                                env=os.environ.copy())  
            if result.returncode == 0:
                return 'ffmpeg'
                
        except Exception as e:
            logger.debug(f"Error while searching for system FFmpeg: {e}")
            
        logger.debug("No system FFmpeg found")
        return None

    def _check_topaz_license(self, topaz_ffmpeg_path):
        """Check if Topaz Video AI license is available"""
        try:
            ffmpeg_exe = os.path.join(topaz_ffmpeg_path, 'ffmpeg.exe')
            if not os.path.exists(ffmpeg_exe):
                return False, f"Topaz FFmpeg not found at {ffmpeg_exe}"

            # Test with a tvai_up filter to check license + model availability.
            # 用默认放大模型 Proteus (prob-4) 做探测，与节点默认一致。
            # 注意: 新版 ffmpeg 8.1 在输入帧数极少 (<=2帧) 时会段错误，
            # 所以这里用足够长的测试源 (1秒, 10帧) 保证稳定。
            test_cmd = [
                ffmpeg_exe, "-f", "lavfi", "-i", "testsrc2=duration=1:size=320x240:rate=10",
                "-vf", "tvai_up=model=prob-4:scale=1.0:device=-1:estimate=0",
                "-f", "null", "-"
            ]

            result = subprocess.run(test_cmd, capture_output=True, text=True, timeout=120,
                                    env=topaz_env_for_subprocess())

            if "floating license not available" in result.stderr:
                return False, "浮动许可证不可用"
            elif "Failed to configure output pad" in result.stderr and "tvai_up" in result.stderr:
                return False, "Topaz滤镜配置失败，可能是许可证或模型问题"
            elif "Model not found" in result.stderr:
                return False, ("Topaz 模型未找到 (模型目录可能指向旧版残留路径)。"
                               "请在节点中确认 Topaz 安装路径，或删除环境变量 "
                               "TVAI_MODEL_DIR / TVAI_MODEL_DATA_DIR 后重启 ComfyUI")
            elif result.returncode == 0:
                return True, "许可证正常"
            else:
                return False, f"未知错误: {result.stderr[:200]}"
                
        except subprocess.TimeoutExpired:
            return False, "许可证检查超时"
        except Exception as e:
            return False, f"许可证检查异常: {str(e)}"

    def _get_topaz_ffmpeg_path(self, ffmpeg_base_path, for_topaz=False, force_topaz=True):
        """Get appropriate FFmpeg path based on context and force_topaz setting"""
        # If force_topaz is True or this is a Topaz-specific operation, use Topaz FFmpeg
        if force_topaz or for_topaz:
            topaz_ffmpeg = os.path.join(ffmpeg_base_path, 'ffmpeg.exe')
            if not os.path.exists(topaz_ffmpeg):
                logger.warning(f"Topaz FFmpeg not found at {topaz_ffmpeg}")
                if not force_topaz:
                    logger.info("Falling back to system FFmpeg")
                    system_ffmpeg = self._get_system_ffmpeg()
                    if system_ffmpeg:
                        return system_ffmpeg
                raise FileNotFoundError(f"FFmpeg not found at {topaz_ffmpeg}")
            return topaz_ffmpeg
            
        # Otherwise, try system FFmpeg first
        system_ffmpeg = self._get_system_ffmpeg()
        if system_ffmpeg:
            return system_ffmpeg
            
        # Fallback to Topaz FFmpeg if system FFmpeg is not available
        logger.warning("System FFmpeg not found, falling back to Topaz FFmpeg")
        return os.path.join(ffmpeg_base_path, 'ffmpeg.exe')

    def _save_batch(self, frames_batch, frame_dir, start_idx):
        """Helper function to save a batch of frames"""
        frame_paths = []
        for i, frame in enumerate(frames_batch):
            frame_path = os.path.join(frame_dir, f"frame_{start_idx + i:05d}.png")
            img = Image.fromarray(frame)
            img.save(frame_path)
            frame_paths.append(frame_path)
        return frame_paths

    def _batch_to_video(self, image_batch, output_path, use_gpu, topaz_ffmpeg_path, force_topaz_ffmpeg, input_fps=24):
        # 优先用 rawvideo pipe 方案: 把图像批次转成 raw RGB24 字节流，通过 stdin 喂给 ffmpeg。
        # 比旧的"存PNG序列再读"快约 3-4 倍 (省掉 PNG 编码+解码+磁盘IO)，
        # 且不产生临时文件。失败时回退到 PNG 序列方案。
        t_start = time.time()
        if self._batch_to_video_pipe(image_batch, output_path, use_gpu, topaz_ffmpeg_path, force_topaz_ffmpeg, input_fps):
            logger.info(f"_batch_to_video (rawvideo pipe) took {time.time()-t_start:.2f}s")
            return
        logger.warning("rawvideo pipe failed, falling back to PNG sequence method")
        self._batch_to_video_png(image_batch, output_path, use_gpu, topaz_ffmpeg_path, force_topaz_ffmpeg, input_fps)
        logger.info(f"_batch_to_video (PNG fallback) took {time.time()-t_start:.2f}s")

    def _batch_to_video_pipe(self, image_batch, output_path, use_gpu, topaz_ffmpeg_path, force_topaz_ffmpeg, input_fps=24):
        """
        用 rawvideo pipe 把图像批次编码成视频。
        image_batch: torch.Tensor (N,H,W,3) float [0,1]。
        通过 stdin 传 raw RGB24 字节给 ffmpeg，省掉 PNG 中间文件。
        成功返回 True，失败返回 False (调用方回退)。
        """
        try:
            import numpy as _np
            # 转 uint8 HWC 连续字节流。无论 GPU/CPU，最终都要 cpu numpy。
            frames = image_batch.detach().cpu().numpy()
            frames = (frames * 255).clip(0, 255).astype(_np.uint8)
            # 确保内存连续 (rawvideo 要求)
            n, h, w, _ = frames.shape
            if not frames.flags['C_CONTIGUOUS']:
                frames = _np.ascontiguousarray(frames)
            raw = frames.tobytes()

            ffmpeg_exe = self._get_topaz_ffmpeg_path(topaz_ffmpeg_path, False, force_topaz_ffmpeg)
            cmd = [
                ffmpeg_exe, "-y", "-hide_banner", "-nostdin",
                # 输入: 从 stdin 读 raw RGB24, 需要指定尺寸和帧率
                "-f", "rawvideo",
                "-pixel_format", "rgb24",
                "-video_size", f"{w}x{h}",
                "-framerate", str(input_fps),
                "-i", "-",
            ]
            cmd.extend(self._get_encoder_args(use_gpu, topaz_ffmpeg_path))
            cmd.extend(["-r", str(input_fps), output_path])

            logger.debug(f"Running FFmpeg rawvideo pipe: {w}x{h} {n}frames {input_fps}fps")
            result = subprocess.run(cmd, input=raw, capture_output=True,
                                    env=topaz_env_for_subprocess(), timeout=600)
            if result.returncode != 0 or not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                err = result.stderr.decode('utf-8', errors='replace')[:300] if result.stderr else ""
                logger.debug(f"rawvideo pipe failed (exit {result.returncode}): {err}")
                return False
            return True
        except Exception as e:
            logger.debug(f"rawvideo pipe exception: {e}")
            return False

    def _batch_to_video_png(self, image_batch, output_path, use_gpu, topaz_ffmpeg_path, force_topaz_ffmpeg, input_fps=24):
        """旧方案: 存 PNG 序列再编码。作为 rawvideo pipe 的回退保留。"""
        device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        
        if use_gpu and torch.cuda.is_available():
            frames = image_batch.to(device)
            frames = (frames * 255).byte()
            frames = frames.cpu().numpy()
        else:
            frames = image_batch.cpu().numpy()
            frames = (frames * 255).astype(np.uint8)
        
        frame_dir = os.path.join(self.output_dir, f"input_frames_{uuid.uuid4()}")
        os.makedirs(frame_dir, exist_ok=True)
        logger.debug(f"Created frame directory: {frame_dir}")
        
        try:
            batch_size = 32
            frame_paths = []
            
            with ThreadPoolExecutor() as executor:
                futures = []
                for i in range(0, len(frames), batch_size):
                    batch = frames[i:i + batch_size]
                    futures.append(
                        executor.submit(self._save_batch, batch, frame_dir, i)
                    )
                
                for future in futures:
                    frame_paths.extend(future.result())
            
            logger.debug(f"Saved {len(frame_paths)} frames")
            
            if not frame_paths:
                raise ValueError("No frames were saved")
            
            ffmpeg_exe = self._get_topaz_ffmpeg_path(topaz_ffmpeg_path, False, force_topaz_ffmpeg)
            cmd = [
                ffmpeg_exe, "-y",
                "-hide_banner",
                "-nostdin",
                "-strict", "2",
                "-hwaccel", "auto",
                "-i", os.path.join(frame_dir, "frame_%05d.png"),
            ]

            cmd.extend(self._get_encoder_args(use_gpu, topaz_ffmpeg_path))

            cmd.extend([
                "-r", str(input_fps),
                output_path
            ])

            logger.debug(f"Running FFmpeg command: {' '.join(cmd)}")
            result = self._run_ffmpeg_with_fallback(cmd, topaz_ffmpeg_path, label="batch_to_video")

            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg error: {result.stderr}")
            if not os.path.exists(output_path):
                raise FileNotFoundError(f"Output video not created: {output_path}")
                
            logger.debug(f"Video created successfully at: {output_path}")
            
        finally:
            shutil.rmtree(frame_dir, ignore_errors=True)

    def _video_to_batch(self, video_path, use_gpu, topaz_ffmpeg_path, force_topaz_ffmpeg):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Input video not found: {video_path}")
        
        frame_dir = os.path.join(self.output_dir, f"output_frames_{uuid.uuid4()}")
        os.makedirs(frame_dir, exist_ok=True)
        logger.debug(f"Created output frame directory: {frame_dir}")
        
        try:
            ffmpeg_exe = self._get_topaz_ffmpeg_path(topaz_ffmpeg_path, False, force_topaz_ffmpeg)
            cmd = [
                ffmpeg_exe, "-y",
                "-i", video_path,
                "-vsync", "0",
                os.path.join(frame_dir, "frame_%05d.png")
            ]
            
            logger.debug(f"Running FFmpeg command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg error: {result.stderr}")
            
            frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.png')])
            logger.debug(f"Found {len(frame_files)} output frames")
            
            if not frame_files:
                raise ValueError(f"No frames extracted from video: {video_path}")
            
            frames = []
            
            if use_gpu and CUPY_AVAILABLE:
                logger.debug("Using CuPy for frame processing")
                with cp.cuda.Device(0):
                    for frame_file in frame_files:
                        frame_path = os.path.join(frame_dir, frame_file)
                        img_np = np.array(Image.open(frame_path))
                        frame_gpu = cp.asarray(img_np)
                        frames.append(cp.asnumpy(frame_gpu))
            else:
                logger.debug("Using CPU for frame processing")
                for frame_file in frame_files:
                    frame_path = os.path.join(frame_dir, frame_file)
                    img = Image.open(frame_path)
                    frame = np.array(img)
                    frames.append(frame)
            
            frames_tensor = torch.from_numpy(np.stack(frames)).float() / 255.0
            logger.debug(f"Created tensor with shape: {frames_tensor.shape}")

            return frames_tensor

        finally:
            shutil.rmtree(frame_dir, ignore_errors=True)

    # ---- 编码器选择 ----
    # 浏览器 <video> 标签普遍支持的 mp4 视频编码是 H.264 (yuv420p)。
    # 旧的 mpeg4 (Part 2) 和 hevc (H.265) 浏览器不能/不能稳定播放，
    # 会导致节点预览区黑屏、视频"播放不出来"。因此统一改用 H.264。
    #
    # Topaz 自带的 ffmpeg 没有 libx264 (纯软编)，但有:
    #   h264_nvenc  - NVIDIA 硬件编码 (N 卡, 质量好速度快)
    #   h264_mf     - Windows MediaFoundation 软件编码 (系统自带, 兼容广)
    #   h264_amf    - AMD 硬件编码
    #   h264_qsv    - Intel QuickSync 硬件编码
    # 编码器探测结果缓存一次，避免每帧都跑 -encoders。
    _encoder_available_cache = None

    def _get_encoder_args(self, use_gpu, topaz_ffmpeg_path):
        """
        返回用于 ffmpeg 命令的编码参数列表 (["-c:v","h264_nvenc",...])。
        统一用 H.264 (浏览器可播放)。use_gpu=True 时优先 h264_nvenc；
        否则用软件 h264_mf。都不可用时回退 mpeg4 (预览无法播放但流程不中断)。
        所有编码都加 +faststart，让浏览器能立即开始播放。
        """
        # 探测所有可用的 H.264 编码器 (缓存一次)
        if self._encoder_available_cache is None:
            available = set()
            ffmpeg_exe = os.path.join(topaz_ffmpeg_path, 'ffmpeg.exe')
            try:
                r = subprocess.run([ffmpeg_exe, '-hide_banner', '-encoders'],
                                   capture_output=True, text=True, timeout=15)
                for line in r.stdout.splitlines():
                    parts = line.split()
                    if len(parts) >= 2 and parts[0].startswith('V'):
                        available.add(parts[1])
            except Exception:
                pass
            self._encoder_available_cache = available
        else:
            available = self._encoder_available_cache

        if use_gpu:
            # GPU 路径: 优先 nvenc
            if 'h264_nvenc' in available:
                return ["-c:v", "h264_nvenc", "-profile:v", "main", "-preset", "fast",
                        "-b:v", "0", "-rc", "vbr", "-cq", "21",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
        # CPU 路径 或 GPU 不可用: 用软件/平台 H.264 (强制不用 nvenc)
        for enc in ('h264_mf', 'h264_amf', 'h264_qsv'):
            if enc in available:
                if enc == 'h264_mf':
                    return ["-c:v", "h264_mf", "-b:v", "5M",
                            "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
                return ["-c:v", enc, "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
        # 都不可用: 回退 mpeg4 (预览无法播放，但流程能跑完)
        logger.warning("No H.264 encoder available; falling back to mpeg4 (browser preview will not play)")
        return ["-c:v", "mpeg4", "-q:v", "2", "-pix_fmt", "yuv420p", "-movflags", "+faststart"]

    # 编码器失败时的错误特征 (硬件编码器对分辨率/像素有限制，如 NVENC 消费级卡最大 4096 宽)
    _ENCODER_FAIL_SIGNATURES = (
        "exceeds",                  # "Width 4608 exceeds 4096"
        "No capable devices found", # NVENC 打不开设备
        "Error while opening encoder",
        "not support",              # "encoder does not support ..."
        "Unsupported",
    )

    def _run_ffmpeg_with_fallback(self, cmd, topaz_ffmpeg_path, label="ffmpeg"):
        """
        执行 ffmpeg 命令；若失败且 stderr 提示编码器问题 (如 NVENC 分辨率超限，
        消费级 N 卡 NVENC 最大支持 4096 宽)，自动把编码参数替换为软件 h264_mf 重试一次。
        返回最终的 CompletedProcess (重试成功则返回重试的结果)。
        调用方仍需检查 returncode / stderr。
        """
        result = subprocess.run(cmd, capture_output=True, text=True,
                                env=topaz_env_for_subprocess())
        if result.returncode == 0:
            return result

        err = result.stderr or ""
        # 只在错误看起来是编码器能力问题时才回退，避免掩盖其它错误
        looks_like_encoder_issue = any(sig in err for sig in self._ENCODER_FAIL_SIGNATURES)
        if not looks_like_encoder_issue:
            return result

        # 定位并替换编码参数段 [-c:v, encoder, ...] 为软件 h264_mf 参数。
        # 编码段一定以 "-c:v" 开头；段内参数都是成对的 (-key value) 或带值选项，
        # 遇到下一个已知的"非编码"ffmpeg 选项即段结束。
        try:
            c_v_idx = cmd.index("-c:v")
        except ValueError:
            return result

        non_encoder_opts = {"-r", "-an", "-shortest", "-vf", "-filter_complex",
                            "-i", "-y", "-hide_banner", "-nostdin", "-strict",
                            "-hwaccel", "-f", "-map", "-ac", "-ar", "-af",
                            "-c:a"}
        post_idx = len(cmd)  # 默认: 编码段一直延伸到命令末尾(输出文件前)
        for i in range(c_v_idx + 1, len(cmd)):
            if cmd[i] in non_encoder_opts:
                post_idx = i
                break

        sw_args = ["-c:v", "h264_mf", "-b:v", "10M",
                   "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
        new_cmd = cmd[:c_v_idx] + sw_args + cmd[post_idx:]

        last_err_line = err.strip().splitlines()[-1] if err.strip() else "unknown"
        logger.warning(
            f"{label}: encoder failed ({last_err_line}); "
            f"retrying with software h264_mf."
        )
        logger.debug(f"Fallback command: {' '.join(new_cmd)}")
        return subprocess.run(new_cmd, capture_output=True, text=True,
                              env=topaz_env_for_subprocess())

    def _probe_video_fps(self, video_path, topaz_ffmpeg_path):
        """
        用 ffprobe 探测视频的真实帧率，返回 float (如 24.0) 或 None。
        用于补帧时获取输入视频的权威帧率，避免依赖可能被设错的 input_fps 参数。
        """
        ffprobe = os.path.join(topaz_ffmpeg_path, 'ffprobe.exe')
        if not os.path.isfile(ffprobe):
            return None
        try:
            # avg_frame_rate 是平均帧率 (总帧数/总时长)，比 r_frame_rate 更可靠
            r = subprocess.run(
                [ffprobe, '-v', 'error', '-select_streams', 'v:0',
                 '-show_entries', 'stream=avg_frame_rate', '-of', 'default=noprint_wrappers=1:nokey=1',
                 video_path],
                capture_output=True, text=True, env=topaz_env_for_subprocess(), timeout=30)
            val = r.stdout.strip()
            # avg_frame_rate 格式是 "24000/1001" 或 "24/1"
            if '/' in val:
                num, den = val.split('/')
                num, den = float(num), float(den)
                if den > 0:
                    return num / den
            else:
                f = float(val)
                if f > 0:
                    return f
        except Exception as e:
            logger.debug(f"probe fps failed: {e}")
        return None

    def _mux_audio(self, video_path, audio, output_path, input_fps, topaz_ffmpeg_path, force_topaz_ffmpeg):
        """
        把 VHS AUDIO ({'waveform': Tensor[1,C,N], 'sample_rate': int}) 合并进无声视频。
        参照 ComfyUI-VideoHelperSuite 的做法:
          - 视频流 -c:v copy 不重编码 (很快)
          - 音频以裸 PCM f32le 从 stdin 喂入
          - apad 把音频填充到视频时长，-shortest 对齐
        返回 True 表示成功，False 表示失败 (调用方会回退到无声视频)。
        """
        try:
            waveform = audio['waveform']
            sample_rate = audio['sample_rate']
        except Exception as e:
            logger.warning(f"Invalid AUDIO input, skipping audio mux: {e}")
            return False

        # waveform 形状通常是 (1, channels, samples)； squeeze 掉 batch 维
        if waveform.dim() == 3:
            channels = waveform.size(1)
        elif waveform.dim() == 2:
            channels = waveform.size(0)
        else:
            logger.warning(f"Unexpected waveform shape {tuple(waveform.shape)}, skipping audio mux")
            return False
        if channels < 1:
            logger.warning("Audio has 0 channels, skipping audio mux")
            return False

        ffmpeg_exe = self._get_topaz_ffmpeg_path(topaz_ffmpeg_path, False, force_topaz_ffmpeg)
        if not ffmpeg_exe or not os.path.isfile(ffmpeg_exe):
            logger.warning("No ffmpeg available for audio mux, skipping")
            return False

        # 估算视频时长，作为 apad 的目标长度 (帧数/fps)。多给 1s 余量。
        try:
            probe = subprocess.run(
                [ffmpeg_exe.replace('ffmpeg.exe', 'ffprobe.exe'), '-v', 'error',
                 '-select_streams', 'v:0', '-show_entries', 'stream=nb_read_frames',
                 '-of', 'csv=p=0', video_path],
                capture_output=True, text=True, env=topaz_env_for_subprocess(), timeout=30)
            nb_frames = int(probe.stdout.strip().splitlines()[0]) if probe.stdout.strip() else 0
        except Exception:
            nb_frames = 0
        # 兜底时长 (秒): 用 nb_frames 估算；拿不到就给一个足够大的值，靠 -shortest 截断
        target_dur = (nb_frames / input_fps + 1) if (nb_frames and input_fps) else 99999

        cmd = [
            ffmpeg_exe, "-y", "-hide_banner", "-nostdin", "-v", "error",
            "-i", video_path,                                   # 输入0: 无声视频
            "-ar", str(int(sample_rate)), "-ac", str(channels), # 输入1: 裸 PCM 音频
            "-f", "f32le", "-i", "-",
            "-c:v", "copy",                                     # 视频不重编码
            "-c:a", "aac", "-b:a", "192k",                      # 音频用 aac
            "-af", f"apad=whole_dur={target_dur}",
            "-shortest",
            "-movflags", "+faststart",
            output_path,
        ]
        logger.debug(f"Audio mux command: {' '.join(cmd[:6])} ... <pcm-stdin>")

        # waveform -> (samples, channels) float32 -> bytes
        try:
            if waveform.dim() == 3:
                audio_data = waveform.squeeze(0).transpose(0, 1).contiguous().numpy().tobytes()
            else:
                audio_data = waveform.transpose(0, 1).contiguous().numpy().tobytes()
        except Exception as e:
            logger.warning(f"Failed to serialize audio waveform: {e}")
            return False

        try:
            result = subprocess.run(cmd, input=audio_data, capture_output=True,
                                    env=topaz_env_for_subprocess(), timeout=300)
        except subprocess.TimeoutExpired:
            logger.warning("Audio mux timed out, keeping video without audio")
            return False
        if result.returncode != 0 or not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            err = result.stderr.decode('utf-8', errors='replace')[:300] if result.stderr else ""
            logger.warning(f"Audio mux failed (exit {result.returncode}): {err}")
            return False
        logger.info(f"Audio muxed successfully -> {output_path}")
        return True

    def process_video(self, images, enable_upscale, upscale_factor, upscale_model,
                     compression, noise, details, halo, blur, preblur,
                     enable_interpolation, input_fps, interpolation_multiplier, interpolation_model, use_gpu, topaz_ffmpeg_path,
                     force_topaz_ffmpeg, save_video=False, filename_prefix="TopazVideo", previous_upscale=None,
                     prompt=None, unique_id=None, audio=None):
        if upscale_model == "thm-2" and upscale_factor != 1.0:
            upscale_factor = 1.0
            logger.warning("thm-2 forces upscale_factor=1.0")

        operation_id = str(uuid.uuid4())
        input_video = os.path.join(self.output_dir, f"{operation_id}_input.mp4")
        intermediate_video = os.path.join(self.output_dir, f"{operation_id}_intermediate.mp4")
        output_video = os.path.join(self.output_dir, f"{operation_id}_output.mp4")
        self.temp_files.extend([input_video, intermediate_video, output_video])

        try:
            t_total_start = time.time()
            # Check Topaz license before processing if Topaz features are enabled
            if enable_upscale or enable_interpolation:
                logger.info("检查Topaz Video AI许可证...")
                t0 = time.time()
                license_ok, license_msg = self._check_topaz_license(topaz_ffmpeg_path)
                logger.info(f"许可证检查 took {time.time()-t0:.2f}s: {license_msg}")
                if not license_ok:
                    raise RuntimeError(
                        f"Topaz Video AI许可证检查失败: {license_msg}\n"
                        "请确保：\n"
                        "1. Topaz Video AI应用程序正在运行\n"
                        "2. 许可证有效且未过期\n"
                        "3. 网络连接正常（如果使用网络许可证）\n"
                        "4. 重启Topaz Video AI应用程序后重试"
                    )
                logger.info(f"许可证检查通过: {license_msg}")

            logger.info(f"Converting image batch to video with input fps {input_fps}...")
            self._batch_to_video(images, input_video, use_gpu, topaz_ffmpeg_path, force_topaz_ffmpeg, input_fps)

            current_input = input_video
            current_output = intermediate_video

            # Modify the upscale logic to always apply filters when enable_upscale is True
            if enable_upscale:
                t0 = time.time()
                all_upscale_params = []
                if previous_upscale:
                    all_upscale_params.extend(previous_upscale)

                # Always add current params when enable_upscale is True
                current_params = {
                    "upscale_factor": upscale_factor,
                    "upscale_model": upscale_model,
                    "compression": compression,
                    "noise": noise,
                    "details": details,
                    "halo": halo,
                    "blur": blur,
                    "preblur": preblur,
                }
                # 兼容旧版工作流: 若上游链表里的 params 仍带 blend (旧版数据)，则忽略该键，
                # 并用默认值补齐缺失的新参数，避免 KeyError。
                normalized = []
                for p in all_upscale_params + [current_params]:
                    np_ = {
                        "upscale_factor": p.get("upscale_factor", 2.0),
                        "upscale_model": p.get("upscale_model", "prob-4"),
                    }
                    for fkey, field, default, _, _, _ in UPSCALE_FILTER_PARAMS:
                        np_[fkey] = p.get(fkey, default)
                    normalized.append(np_)

                upscale_filters = []
                for params in normalized:
                    seg = (
                        f"tvai_up=model={params['upscale_model']}"
                        f":scale={params['upscale_factor']}"
                        f":estimate=8"
                    )
                    # 只追加非默认的细分参数，保持滤镜串可读且避免无意义覆盖
                    for fkey, field, default, _, _, _ in UPSCALE_FILTER_PARAMS:
                        val = params[fkey]
                        seg += f":{fkey}={val}"
                    upscale_filters.append(seg)

                filter_chain = ','.join(upscale_filters)
                logger.info(f"Applying upscale filter chain: {filter_chain}")
                ffmpeg_exe = self._get_topaz_ffmpeg_path(topaz_ffmpeg_path, True, force_topaz_ffmpeg)
                cmd = [
                    ffmpeg_exe, "-y",
                    "-hide_banner",
                    "-nostdin",
                    "-strict", "2",
                    "-hwaccel", "auto",
                    "-i", current_input,
                    "-vf", filter_chain,
                ]

                cmd.extend(self._get_encoder_args(use_gpu, topaz_ffmpeg_path))

                cmd.extend([
                    "-r", str(input_fps),
                    current_output
                ])

                logger.debug(f"Running FFmpeg upscale command: {' '.join(cmd)}")
                # 注入修正后的 TVAI_MODEL_DIR，避免旧版残留环境变量导致 'Model not found'。
                # 用带回退的执行封装: 若硬件编码器因分辨率超限等失败，自动改用软件 h264_mf 重试。
                result = self._run_ffmpeg_with_fallback(cmd, topaz_ffmpeg_path, label="upscale")

                if result.returncode != 0:
                    error_msg = result.stderr
                    # Check for specific Topaz license issues
                    if "floating license not available" in error_msg:
                        raise RuntimeError(
                            "Topaz Video AI许可证错误：浮动许可证不可用\n"
                            "解决方案：\n"
                            "1. 确保Topaz Video AI应用程序正在运行\n"
                            "2. 检查许可证是否有效且未过期\n"
                            "3. 如果使用网络许可证，检查网络连接\n"
                            "4. 重启Topaz Video AI应用程序\n"
                            f"原始错误：{error_msg}"
                        )
                    elif "Failed to configure output pad" in error_msg and "tvai_up" in error_msg:
                        raise RuntimeError(
                            "Topaz Video AI滤镜配置失败\n"
                            "可能的原因：\n"
                            "1. 许可证问题 - 确保Topaz Video AI许可证有效\n"
                            "2. 模型不支持 - 检查所选的升频模型是否可用\n"
                            "3. 参数不兼容 - 检查升频参数设置\n"
                            f"原始错误：{error_msg}"
                        )
                    elif "Model not found" in error_msg:
                        raise RuntimeError(
                            "Topaz 模型未找到。这通常是因为模型目录指向了旧版 Topaz Video AI 的残留路径。\n"
                            "解决方法：在节点里确认 Topaz 安装路径正确，或删除环境变量 TVAI_MODEL_DIR / TVAI_MODEL_DATA_DIR\n"
                            "让插件自动探测新版路径 (C:\\ProgramData\\Topaz Labs LLC\\Topaz Video\\models)。\n"
                            f"原始错误：{error_msg}"
                        )
                    else:
                        raise RuntimeError(f"FFmpeg upscale error: {error_msg}")

                logger.info(f"Upscale took {time.time()-t0:.2f}s")
                current_input = current_output
                current_output = output_video
            
            if enable_interpolation:
                # 补帧的目标 fps = 输入视频真实帧率 × 倍数。
                # 优先用 ffprobe 探测 current_input 的真实帧率 (权威值)，
                # 探测失败才回退到用户填的 input_fps 参数。
                real_fps = self._probe_video_fps(current_input, topaz_ffmpeg_path)
                base_fps = real_fps if real_fps and real_fps > 0 else float(input_fps)
                target_fps = int(round(base_fps * interpolation_multiplier))
                logger.info(f"Applying interpolation: base fps={base_fps} (real={real_fps}, param={input_fps}) "
                            f"multiplier={interpolation_multiplier} target fps={target_fps}")
                if target_fps <= 0:
                    raise ValueError(
                        f"补帧目标帧率必须大于 0 (当前 target_fps={target_fps}, "
                        f"base_fps={base_fps}, multiplier={interpolation_multiplier})。"
                        "请检查节点上的 input_fps 和 interpolation_multiplier 设置。"
                    )

                interpolation_filter = f"tvai_fi=model={interpolation_model}:fps={target_fps}"
                
                ffmpeg_exe = self._get_topaz_ffmpeg_path(topaz_ffmpeg_path, True, force_topaz_ffmpeg)
                cmd = [
                    ffmpeg_exe, "-y",
                    "-hide_banner",
                    "-nostdin",
                    "-strict", "2",
                    "-hwaccel", "auto",
                    "-i", current_input,
                    "-vf", interpolation_filter,
                ]

                cmd.extend(self._get_encoder_args(use_gpu, topaz_ffmpeg_path))

                cmd.extend([
                    current_output
                ])

                logger.debug(f"Running FFmpeg interpolation command: {' '.join(cmd)}")
                result = self._run_ffmpeg_with_fallback(cmd, topaz_ffmpeg_path, label="interpolation")

                if result.returncode != 0:
                    error_msg = result.stderr
                    # Check for specific Topaz license issues
                    if "floating license not available" in error_msg:
                        raise RuntimeError(
                            "Topaz Video AI许可证错误：浮动许可证不可用\n"
                            "解决方案：\n"
                            "1. 确保Topaz Video AI应用程序正在运行\n"
                            "2. 检查许可证是否有效且未过期\n"
                            "3. 如果使用网络许可证，检查网络连接\n"
                            "4. 重启Topaz Video AI应用程序\n"
                            f"原始错误：{error_msg}"
                        )
                    elif "Failed to configure output pad" in error_msg and "tvai_fi" in error_msg:
                        raise RuntimeError(
                            "Topaz Video AI插值滤镜配置失败\n"
                            "可能的原因：\n"
                            "1. 许可证问题 - 确保Topaz Video AI许可证有效\n"
                            "2. 插值模型不支持 - 检查所选的插值模型是否可用\n"
                            "3. FPS参数不兼容 - 检查目标FPS设置\n"
                            f"原始错误：{error_msg}"
                        )
                    elif "Model not found" in error_msg:
                        raise RuntimeError(
                            "Topaz 模型未找到。模型目录可能指向了旧版残留路径。\n"
                            "解决方法：确认 Topaz 安装路径正确，或删除环境变量 TVAI_MODEL_DIR / TVAI_MODEL_DATA_DIR\n"
                            "让插件自动探测新版路径。\n"
                            f"原始错误：{error_msg}"
                        )
                    else:
                        raise RuntimeError(f"FFmpeg interpolation error: {error_msg}")
            else:
                if current_input != output_video:
                    shutil.copy2(current_input, current_output)

            # ---- 输出阶段 ----
            # 设计原则: Topaz 处理后已经是视频，默认直接在节点上预览该视频，
            # 不再无谓地解码回图像序列。只有当 IMAGE 端口真的被下游节点连接时，
            # 才执行 (昂贵的) 视频->图像批次解码，供下游继续处理。
            #
            # 无论 save_video 与否，前端预览器都需要一个落盘的 mp4 文件才能播放:
            #   - save_video=True : 按 filename_prefix 命名，持久留在 output 目录
            #   - save_video=False: 写入 temp 目录供预览 (用户视为"不保存"，不污染 output)
            if save_video:
                output_dir = folder_paths.get_output_directory()
                full_output_folder, filename, _, subfolder, _ = folder_paths.get_save_image_path(filename_prefix, output_dir)
                # 持久保存: 查找下一个可用计数器，避免覆盖已有文件
                max_counter = 0
                matcher = re.compile(f"{re.escape(filename)}_(\\d+)\\D*\\..+", re.IGNORECASE)
                for existing_file in os.listdir(full_output_folder):
                    match = matcher.fullmatch(existing_file)
                    if match:
                        file_counter = int(match.group(1))
                        if file_counter > max_counter:
                            max_counter = file_counter
                counter = max_counter + 1
                output_file = f"{filename}_{counter:05}.mp4"
                output_path = os.path.join(full_output_folder, output_file)
                preview_type = "output"
                preview_subfolder = subfolder
            else:
                # 仅用于预览: 写入 ComfyUI temp 目录，type=temp 让前端从 temp 取
                full_output_folder = folder_paths.get_temp_directory()
                output_file = f"{filename_prefix}_preview_{uuid.uuid4().hex[:8]}.mp4"
                output_path = os.path.join(full_output_folder, output_file)
                preview_type = "temp"
                preview_subfolder = ""

            # 落盘最终视频。如果接了音频，用 ffmpeg 把音轨 mux 进去 (视频流不重编码)；
            # 否则直接复制无声视频。音频合成失败时回退到无声视频，保证不中断流程。
            if audio is not None:
                logger.info("Audio input detected, muxing audio into output video...")
                t0 = time.time()
                muxed_ok = self._mux_audio(current_output, audio, output_path, input_fps,
                                           topaz_ffmpeg_path, force_topaz_ffmpeg)
                logger.info(f"Audio mux took {time.time()-t0:.2f}s")
                if not muxed_ok:
                    logger.warning("Audio mux failed or skipped; falling back to video without audio")
                    shutil.copy2(current_output, output_path)
            else:
                shutil.copy2(current_output, output_path)
            logger.info(f"Output video ({preview_type}): {output_path}")
            logger.info(f"process_video total took {time.time()-t_total_start:.2f}s")

            # 前端预览信息 (ComfyUI 前端用 gifs 字段渲染视频预览)
            previews = [{
                "filename": output_file,
                "subfolder": preview_subfolder,
                "type": preview_type,
                "format": "video/mp4",
                "frame_rate": input_fps,
            }]

            # 判断 IMAGE 端口是否被下游连接，决定要不要解码
            image_connected = self._is_image_output_connected(prompt, unique_id)

            if image_connected:
                # 下游接了 IMAGE 节点，必须解码回图像批次
                logger.info("IMAGE output is connected to downstream; decoding video back to image batch...")
                output_frames = self._video_to_batch(current_output, use_gpu, topaz_ffmpeg_path, force_topaz_ffmpeg)
                return {"ui": {"gifs": previews}, "result": (output_frames,)}
            else:
                # 没有下游消费 IMAGE：跳过解码，直接返回原始输入图像占位
                # (返回 images 而非 None，保证 result 形状与 RETURN_TYPES 一致，不断图)
                logger.info("No downstream IMAGE consumer; skipping decode, previewing video directly.")
                return {"ui": {"gifs": previews}, "result": (images,)}
            
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            raise

NODE_CLASS_MAPPINGS = {
    "TopazVideoAI": TopazVideoAINode,
    "TopazUpscaleParams": TopazUpscaleParamsNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TopazVideoAI": "Topaz Video AI (Upscale & Frame Interpolation)",
    "TopazUpscaleParams": "Topaz Upscale Parameters"
}
