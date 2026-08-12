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

# ---------------------------------------------------------------------------
# 服务端端点: 让前端能预览任意路径的视频文件 (LoadVideoFFmpegPath 需要)
# ComfyUI 原生 /view 只支持 input/output/temp 目录，不支持任意路径。
# ---------------------------------------------------------------------------
import os
import mimetypes
try:
    from server import PromptServer
    from aiohttp import web

    _VIDEO_EXTS = {'.mp4', '.webm', '.mkv', '.mov', '.avi', '.gif', '.flv', '.ts', '.m4v', '.mpg', '.mpeg'}

    @PromptServer.instance.routes.get("/topaz/view_video")
    async def topaz_view_video(request):
        """预览任意路径的视频文件。仅允许视频扩展名，防止误暴露其它文件。"""
        path = request.rel_url.query.get("filename", "")
        if not path:
            return web.Response(status=400, text="missing filename")
        path = os.path.normpath(path)
        ext = os.path.splitext(path)[1].lower()
        if ext not in _VIDEO_EXTS:
            return web.Response(status=403, text="not a video file")
        if not os.path.isfile(path):
            return web.Response(status=404, text="file not found")
        ct = mimetypes.guess_type(path)[0] or 'application/octet-stream'
        return web.FileResponse(path, headers={
            "Content-Type": ct,
            "Cache-Control": "no-store",
        })
except ImportError:
    pass

# 前端扩展目录: 让 ComfyUI 加载 web/js/ 下的 JS，为节点添加视频预览 widget
WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
