// ComfyUI-TopazVideoAI 前端扩展
//
// 给以下节点添加视频预览 widget:
//   TopazVideoAI            - 处理完后从后端 gifs 数据显示视频
//   TopazLoadVideoFFmpeg     - 选了文件后实时预览 (input 目录, 用 /view)
//   TopazLoadVideoFFmpegPath - 选了文件后实时预览 (任意路径, 用 /topaz/view_video)
//
// ComfyUI 原生前端不会为 mp4 创建 <video> 播放器，必须由插件 JS 自行实现。

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const PREFIX = "[TopazVideoAI]";

function chainCallback(object, property, callback) {
    if (object == undefined) return;
    if (property in object && object[property]) {
        const orig = object[property];
        object[property] = function () {
            const r = orig.apply(this, arguments);
            return callback.apply(this, arguments) ?? r;
        };
    } else {
        object[property] = callback;
    }
}

function fitHeight(node) {
    node.setSize([node.size[0], node.computeSize([node.size[0], node.size[1]])[1]]);
    node?.graph?.setDirtyCanvas(true, true);
}

// ---------------------------------------------------------------------------
// 通用: 在节点上创建 videopreview widget (含 <video> 元素)
// 返回 previewWidget，调用方负责后续的更新逻辑。
// ---------------------------------------------------------------------------
function createVideoWidget(previewNode) {
    const element = document.createElement("div");
    const previewWidget = previewNode.addDOMWidget("videopreview", "preview", element, {
        serialize: false,
        hideOnZoom: false,
        getValue() { return element.value; },
        setValue(v) { element.value = v; },
    });

    previewWidget.computeSize = function (width) {
        if (this.aspectRatio && !this.parentEl?.hidden) {
            let height = (previewNode.size[0] - 20) / this.aspectRatio + 10;
            if (!(height > 0)) height = 0;
            this.computedHeight = height + 10;
            return [width, this.computedHeight];
        }
        return [width, -4];
    };

    previewWidget.value = { hidden: false, paused: false, params: {}, muted: true };

    previewWidget.parentEl = document.createElement("div");
    previewWidget.parentEl.className = "topaz_preview";
    previewWidget.parentEl.style.width = "100%";
    element.appendChild(previewWidget.parentEl);

    previewWidget.videoEl = document.createElement("video");
    previewWidget.videoEl.controls = false;
    previewWidget.videoEl.loop = true;
    previewWidget.videoEl.muted = true;
    previewWidget.videoEl.preload = "auto";
    previewWidget.videoEl.setAttribute("playsinline", "");
    previewWidget.videoEl.style.width = "100%";
    // 提升到独立合成层，避免和 litegraph 画布逐帧合成导致掉帧
    previewWidget.videoEl.style.willChange = "transform";
    previewWidget.videoEl.style.transform = "translateZ(0)";
    previewWidget.videoEl.addEventListener("mouseenter", () => {
        previewWidget.videoEl.controls = true;
    });
    previewWidget.videoEl.addEventListener("mouseleave", () => {
        previewWidget.videoEl.controls = false;
    });
    previewWidget.videoEl.addEventListener("loadedmetadata", () => {
        if (previewWidget.videoEl.videoWidth && previewWidget.videoEl.videoHeight) {
            previewWidget.aspectRatio =
                previewWidget.videoEl.videoWidth / previewWidget.videoEl.videoHeight;
        }
        fitHeight(previewNode);
    });
    previewWidget.videoEl.addEventListener("error", () => {
        const code = previewWidget.videoEl.error?.code;
        const msg = previewWidget.videoEl.error?.message;
        console.warn(PREFIX, "video error", "code=", code, "msg=", msg,
            "src=", previewWidget.videoEl.src);
        previewWidget.parentEl.hidden = true;
        fitHeight(previewNode);
    });
    previewWidget.parentEl.appendChild(previewWidget.videoEl);

    // 统一的 src 更新: 根据 params 选择正确的端点
    previewWidget.updateSource = function () {
        const params = this.value?.params;
        if (!params?.filename) return;
        const q = Object.assign({}, params);
        q.timestamp = Date.now();
        this.parentEl.hidden = this.value.hidden ?? false;
        const fmt = String(q.format || "");
        if (fmt.split("/")[0] !== "video") {
            this.videoEl.hidden = true;
            return;
        }
        // Path 版任意路径用 /topaz/view_video, input/output/temp 用原生 /view
        let url;
        if (q._arbitrary_path) {
            delete q._arbitrary_path;
            delete q.type;
            url = api.apiURL("/topaz/view_video?" + new URLSearchParams({ filename: params.filename }));
        } else {
            url = api.apiURL("/view?" + new URLSearchParams(q));
        }
        this.videoEl.autoplay = !this.value.paused && !this.value.hidden;
        this.videoEl.src = url;
        this.videoEl.hidden = false;
        this.videoEl.load();
    };
    previewWidget.callback = previewWidget.updateSource;
    return previewWidget;
}

// ---------------------------------------------------------------------------
// TopazVideoAI 节点: onExecuted 收后端 gifs 数据更新预览
// ---------------------------------------------------------------------------
function addVideoPreview(nodeType) {
    chainCallback(nodeType.prototype, "onNodeCreated", function () {
        createVideoWidget(this);
    });
    chainCallback(nodeType.prototype, "onExecuted", function (message) {
        if (message?.gifs?.length) {
            const w = this.widgets?.find((w) => w.name === "videopreview");
            if (w) {
                w.value.params = message.gifs[0];
                w.updateSource();
            }
        }
    });
}

// ---------------------------------------------------------------------------
// LoadVideo 节点: 监听 video widget 变化，实时预览 (不需要执行工作流)
// ---------------------------------------------------------------------------
function addLoadVideoPreview(nodeType, isPath) {
    chainCallback(nodeType.prototype, "onNodeCreated", function () {
        const previewNode = this;
        const previewWidget = createVideoWidget(previewNode);

        // 监听 video widget 变化
        const videoWidget = this.widgets?.find((w) => w.name === "video");
        if (!videoWidget) return;

        chainCallback(videoWidget, "callback", function (value) {
            if (!value) return;
            let ext = value.split(".").pop().toLowerCase();
            let format = ["gif", "webp", "avif"].includes(ext) ? "image" : "video";
            format += "/" + ext;

            if (isPath) {
                // Path 版: 任意路径，用 /topaz/view_video 端点
                previewWidget.value.params = {
                    filename: value,
                    format: format,
                    _arbitrary_path: true,
                };
            } else {
                // Upload 版: 文件在 input 目录，用原生 /view
                previewWidget.value.params = {
                    filename: value,
                    type: "input",
                    format: format,
                };
            }
            previewWidget.updateSource();
        });

        // 初始触发: 如果已有值 (如加载工作流时) 立即预览
        if (videoWidget.value) {
            try { videoWidget.callback(videoWidget.value); } catch (e) {}
        }
    });
}

// ---------------------------------------------------------------------------
// 注册
// ---------------------------------------------------------------------------
app.registerExtension({
    name: "ComfyUI.TopazVideoAI",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData?.name === "TopazVideoAI") {
            addVideoPreview(nodeType);
        } else if (nodeData?.name === "TopazLoadVideoFFmpeg") {
            addLoadVideoPreview(nodeType, false);  // Upload 版
        } else if (nodeData?.name === "TopazLoadVideoFFmpegPath") {
            addLoadVideoPreview(nodeType, true);   // Path 版
        }
    },
});
