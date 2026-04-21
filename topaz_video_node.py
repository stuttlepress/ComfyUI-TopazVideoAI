import os
import re
import json
import glob
import winreg
import numpy as np
import torch
import subprocess
import uuid
from PIL import Image
import logging
import shutil
from concurrent.futures import ThreadPoolExecutor
import folder_paths

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('TopazVideoAI')


def _topaz_model_dir():
    return os.path.join(
        os.environ.get("PROGRAMDATA", r"C:\ProgramData"),
        r"Topaz Labs LLC\Topaz Video\models"
    )


def _topaz_data_dir():
    try:
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Topaz Labs LLC\Topaz Video")
        data_dir, _ = winreg.QueryValueEx(key, "veaiDataFolder")
        winreg.CloseKey(key)
        return data_dir
    except Exception:
        return _topaz_model_dir()


def _downloaded_prefixes(data_dir):
    """Return set of 'shortname-vversion' prefixes that have at least one .tz3 file on disk."""
    prefixes = set()
    try:
        for f in os.listdir(data_dir):
            if f.endswith('.tz3'):
                parts = f.split('-')
                if len(parts) >= 2:
                    prefixes.add(f"{parts[0]}-{parts[1]}")
    except Exception:
        pass
    return prefixes


def _discover_models():
    """
    Returns (upscale_models, interpolation_models).
    Each list contains model IDs for downloaded models first, then
    'model-id [not downloaded]' for the rest. Both groups are sorted alphabetically.
    Falls back to hardcoded lists if the model directory cannot be read.
    """
    model_dir = _topaz_model_dir()
    data_dir = _topaz_data_dir()
    downloaded = _downloaded_prefixes(data_dir)

    upscale = []
    interpolation = []

    try:
        for json_path in sorted(glob.glob(os.path.join(model_dir, "*.json"))):
            try:
                with open(json_path) as f:
                    d = json.load(f)
            except Exception:
                continue

            if not isinstance(d, dict):
                continue

            model_type = d.get("modelType")
            if model_type not in (1, 2):
                continue
            if not d.get("enabled", 1):
                continue

            name = os.path.basename(json_path)[:-5]
            short_name = d.get("shortName", "")
            version = d.get("version", "")
            is_downloaded = f"{short_name}-v{version}" in downloaded

            entry = (name, is_downloaded)
            if model_type == 1:
                upscale.append(entry)
            else:
                interpolation.append(entry)
    except Exception:
        pass

    def latest_per_family(entries):
        families = {}
        for name, is_downloaded in entries:
            m = re.match(r'^(.+)-(\d+)$', name)
            if m:
                family, version = m.group(1), int(m.group(2))
            else:
                family, version = name, 0
            if family not in families or version > families[family][0]:
                families[family] = (version, name, is_downloaded)
        return [(name, ok) for _, name, ok in families.values()]

    def build_list(entries):
        entries = latest_per_family(entries)
        ready = sorted(name for name, ok in entries if ok)
        missing = sorted(f"{name} [not downloaded]" for name, ok in entries if not ok)
        return ready + missing

    upscale_list = build_list(upscale)
    interpolation_list = build_list(interpolation)

    if not upscale_list:
        upscale_list = ["aaa-9", "ahq-12", "alq-13", "alqs-2", "amq-13", "amqs-2",
                        "ghq-5", "iris-2", "iris-3", "nyx-3", "prob-4", "thf-4",
                        "thd-3", "thm-2", "rhea-1", "rxl-1"]
    if not interpolation_list:
        interpolation_list = ["apo-8", "apf-1", "chr-2", "chf-3"]

    return upscale_list, interpolation_list


def _model_id(name):
    """Strip the '[not downloaded]' suffix to get the bare model ID for ffmpeg."""
    return name.split(' [')[0]


_UPSCALE_MODELS, _INTERPOLATION_MODELS = _discover_models()


class TopazUpscaleParamsNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "upscale_factor": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.5}),
                "upscale_model": (_UPSCALE_MODELS, {"default": _UPSCALE_MODELS[0]}),
                "compression": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 1.0, "step": 0.1}),
                "blend": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.1}),
            },
            "optional": {
                "previous_upscale": ("UPSCALE_PARAMS",),
            }
        }

    RETURN_TYPES = ("UPSCALE_PARAMS",)
    FUNCTION = "get_params"
    CATEGORY = "video"

    def get_params(self, upscale_factor=2.0, upscale_model=None, compression=1.0, blend=0.0, previous_upscale=None):
        if upscale_model is None:
            upscale_model = _UPSCALE_MODELS[0]
        model_id = _model_id(upscale_model)

        if model_id == "thm-2" and upscale_factor != 1.0:
            upscale_factor = 1.0
            logger.warning("thm-2 forces upscale_factor=1.0")

        current_params = {
            "upscale_factor": upscale_factor,
            "upscale_model": model_id,
            "compression": compression,
            "blend": blend
        }

        if previous_upscale is None:
            return ([current_params],)
        else:
            return (previous_upscale + [current_params],)


class TopazVideoAINode:
    def __init__(self):
        self.output_dir = os.path.join(folder_paths.get_temp_directory(), "topaz")
        os.makedirs(self.output_dir, exist_ok=True)
        logger.debug(f"Initialized temp directory at: {self.output_dir}")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "enable_upscale": ("BOOLEAN", {"default": False}),
                "upscale_factor": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.5}),
                "upscale_model": (_UPSCALE_MODELS, {"default": _UPSCALE_MODELS[0]}),
                "compression": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 1.0, "step": 0.1}),
                "blend": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.1}),
                "enable_interpolation": ("BOOLEAN", {"default": False}),
                "input_fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.001}),
                "interpolation_multiplier": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 8.0, "step": 0.5}),
                "interpolation_mode": (["target_fps", "multiplier"], {"default": "target_fps"}),
                "target_fps": ("FLOAT", {"default": 48.0, "min": 1.0, "max": 960.0, "step": 0.001}),
                "interpolation_model": (_INTERPOLATION_MODELS, {"default": _INTERPOLATION_MODELS[0]}),
                "topaz_ffmpeg_path": ("STRING", {"default": os.path.join(os.environ.get("PROGRAMFILES", r"C:\Program Files"), r"Topaz Labs LLC\Topaz Video")}),
            },
            "optional": {
                "previous_upscale": ("UPSCALE_PARAMS",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    OUTPUT_NODE = True
    FUNCTION = "process_video"
    CATEGORY = "video"

    def _get_topaz_ffmpeg_path(self, ffmpeg_base_path):
        ffmpeg_exe = os.path.join(ffmpeg_base_path, 'ffmpeg.exe')
        if not os.path.exists(ffmpeg_exe):
            raise FileNotFoundError(f"Topaz FFmpeg not found at {ffmpeg_exe}")
        return ffmpeg_exe

    def _save_batch(self, frames_batch, frame_dir, start_idx):
        frame_paths = []
        for i, frame in enumerate(frames_batch):
            frame_path = os.path.join(frame_dir, f"frame_{start_idx + i:05d}.png")
            img = Image.fromarray(frame)
            img.save(frame_path)
            frame_paths.append(frame_path)
        return frame_paths

    def _topaz_env(self):
        env = os.environ.copy()
        env["TVAI_MODEL_DIR"] = _topaz_model_dir()
        env["TVAI_MODEL_DATA_DIR"] = _topaz_data_dir()
        logger.warning(f"TVAI_MODEL_DIR={env['TVAI_MODEL_DIR']} TVAI_MODEL_DATA_DIR={env['TVAI_MODEL_DATA_DIR']}")
        return env

    def _batch_to_video(self, image_batch, output_path, topaz_ffmpeg_path, input_fps=24):
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

            ffmpeg_exe = self._get_topaz_ffmpeg_path(topaz_ffmpeg_path)
            cmd = [
                ffmpeg_exe, "-y",
                "-hide_banner",
                "-nostdin",
                "-strict", "2",
                "-hwaccel", "auto",
                "-i", os.path.join(frame_dir, "frame_%05d.png"),
                "-c:v", "ffv1",
                "-pix_fmt", "rgb24",
                "-r", str(input_fps),
                output_path
            ]

            logger.debug(f"Running FFmpeg command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=topaz_ffmpeg_path, env=self._topaz_env())

            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg error: {result.stderr}")

            if not os.path.exists(output_path):
                raise FileNotFoundError(f"Output video not created: {output_path}")

            logger.debug(f"Video created successfully at: {output_path}")

        finally:
            shutil.rmtree(frame_dir, ignore_errors=True)

    def _video_to_batch(self, video_path, topaz_ffmpeg_path):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Input video not found: {video_path}")

        frame_dir = os.path.join(self.output_dir, f"output_frames_{uuid.uuid4()}")
        os.makedirs(frame_dir, exist_ok=True)
        logger.debug(f"Created output frame directory: {frame_dir}")

        try:
            ffmpeg_exe = self._get_topaz_ffmpeg_path(topaz_ffmpeg_path)
            cmd = [
                ffmpeg_exe, "-y",
                "-i", video_path,
                "-vsync", "0",
                os.path.join(frame_dir, "frame_%05d.png")
            ]

            logger.debug(f"Running FFmpeg command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=topaz_ffmpeg_path, env=self._topaz_env())

            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg error: {result.stderr}")

            frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.png')])
            logger.debug(f"Found {len(frame_files)} output frames")

            if not frame_files:
                raise ValueError(f"No frames extracted from video: {video_path}")

            frames = []
            for frame_file in frame_files:
                frame_path = os.path.join(frame_dir, frame_file)
                frames.append(np.array(Image.open(frame_path)))

            frames_tensor = torch.from_numpy(np.stack(frames)).float() / 255.0
            logger.debug(f"Created tensor with shape: {frames_tensor.shape}")

            return frames_tensor

        finally:
            shutil.rmtree(frame_dir, ignore_errors=True)

    def process_video(self, images, enable_upscale, upscale_factor, upscale_model, compression, blend,
                     enable_interpolation, input_fps, interpolation_multiplier,
                     interpolation_mode, target_fps,
                     interpolation_model, topaz_ffmpeg_path,
                     previous_upscale=None):
        upscale_id = _model_id(upscale_model)
        interpolation_id = _model_id(interpolation_model)

        if upscale_id == "thm-2" and upscale_factor != 1.0:
            upscale_factor = 1.0
            logger.warning("thm-2 forces upscale_factor=1.0")

        operation_id = str(uuid.uuid4())
        input_video = os.path.join(self.output_dir, f"{operation_id}_input.mp4")
        intermediate_video = os.path.join(self.output_dir, f"{operation_id}_intermediate.mp4")
        output_video = os.path.join(self.output_dir, f"{operation_id}_output.mp4")
        try:
            logger.info(f"Converting image batch to video with input fps {input_fps}...")
            self._batch_to_video(images, input_video, topaz_ffmpeg_path, input_fps)

            current_input = input_video
            current_output = intermediate_video

            if enable_upscale:
                all_upscale_params = []
                if previous_upscale:
                    all_upscale_params.extend(previous_upscale)

                all_upscale_params.append({
                    "upscale_factor": upscale_factor,
                    "upscale_model": upscale_id,
                    "compression": compression,
                    "blend": blend
                })

                upscale_filters = []
                for params in all_upscale_params:
                    upscale_filters.append(
                        f"tvai_up=model={params['upscale_model']}"
                        f":scale={int(params['upscale_factor'])}"
                        f":estimate=8"
                        f":compression={params['compression']}"
                        f":blend={params['blend']}"
                    )

                filter_chain = ','.join(upscale_filters)
                logger.info(f"Applying upscale filter chain: {filter_chain}")
                ffmpeg_exe = self._get_topaz_ffmpeg_path(topaz_ffmpeg_path)
                cmd = [
                    ffmpeg_exe, "-y",
                    "-hide_banner",
                    "-nostdin",
                    "-strict", "2",
                    "-hwaccel", "auto",
                    "-i", current_input,
                    "-vf", filter_chain,
                    "-c:v", "ffv1",
                    "-pix_fmt", "rgb24",
                    "-r", str(input_fps),
                    current_output
                ]

                logger.debug(f"Running FFmpeg upscale command: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=topaz_ffmpeg_path, env=self._topaz_env())

                if result.returncode != 0:
                    raise RuntimeError(f"FFmpeg upscale error: {result.stderr}")

                current_input = current_output
                current_output = output_video

            if enable_interpolation:
                if interpolation_mode == "target_fps":
                    logger.info(f"Applying interpolation with direct target fps {target_fps}")
                else:
                    target_fps = input_fps * interpolation_multiplier
                    logger.info(f"Applying interpolation with input fps {input_fps} and multiplier {interpolation_multiplier} (target fps: {target_fps})")
                if target_fps <= 0:
                    raise ValueError("Target FPS must be greater than 0")

                interpolation_filter = f"tvai_fi=model={interpolation_id}:fps={target_fps}"

                ffmpeg_exe = self._get_topaz_ffmpeg_path(topaz_ffmpeg_path)
                cmd = [
                    ffmpeg_exe, "-y",
                    "-hide_banner",
                    "-nostdin",
                    "-strict", "2",
                    "-hwaccel", "auto",
                    "-i", current_input,
                    "-vf", interpolation_filter,
                    "-c:v", "ffv1",
                    "-pix_fmt", "rgb24",
                    current_output
                ]

                logger.debug(f"Running FFmpeg interpolation command: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=topaz_ffmpeg_path, env=self._topaz_env())

                if result.returncode != 0:
                    raise RuntimeError(f"FFmpeg interpolation error: {result.stderr}")
            else:
                if current_input != output_video:
                    shutil.copy2(current_input, current_output)

            logger.info("Converting final video back to image batch...")
            output_frames = self._video_to_batch(current_output, topaz_ffmpeg_path)
            return (output_frames,)

        except Exception as e:
            logger.error(f"An error occurred: {e}")
            raise
        finally:
            for f in [input_video, intermediate_video, output_video]:
                try:
                    os.remove(f)
                except OSError:
                    pass


NODE_CLASS_MAPPINGS = {
    "TopazVideoAI": TopazVideoAINode,
    "TopazUpscaleParams": TopazUpscaleParamsNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TopazVideoAI": "Topaz Video AI (Upscale & Frame Interpolation)",
    "TopazUpscaleParams": "Topaz Upscale Parameters"
}
