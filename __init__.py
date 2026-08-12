from .topaz_video_node import TopazVideoAINode,TopazUpscaleParamsNode
from .video_load import TopazLoadVideoFFmpeg, TopazLoadVideoFFmpegPath

# 定义节点类映射
NODE_CLASS_MAPPINGS = {
    "TopazVideoAI": TopazVideoAINode,
    "TopazUpscaleParams": TopazUpscaleParamsNode,
    "TopazLoadVideoFFmpeg": TopazLoadVideoFFmpeg,
    "TopazLoadVideoFFmpegPath": TopazLoadVideoFFmpegPath,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TopazVideoAI": "Topaz Video AI (Upscale & Frame Interpolation)",
    "TopazUpscaleParams": "Topaz Upscale Parameters",
    "TopazLoadVideoFFmpeg": "Load Video FFmpeg (Topaz)",
    "TopazLoadVideoFFmpegPath": "Load Video Path FFmpeg (Topaz)",
}

# 前端扩展目录: 让 ComfyUI 加载 web/js/ 下的 JS，为节点添加视频预览 widget
WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
