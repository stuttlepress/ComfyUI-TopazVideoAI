// ComfyUI-TopazVideoAI 前端扩展
//
// 作用: 给 TopazVideoAI 节点添加视频预览 widget。
//
// 背景: ComfyUI 原生前端只会把 ui.gifs 里的内容当作图片/GIF 渲染，
// 不会为 mp4 文件创建 <video> 播放器。视频预览必须由自定义节点的
// 前端扩展自行实现 (ComfyUI-VideoHelperSuite 也是这么做的)。
//
// 本文件用原生 /view 端点取视频 (不依赖 VHS 的 /vhs/viewvideo)，
// 因此即使没装 VHS 也能独立工作。

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const PREFIX = "[TopazVideoAI]";

// 把 callback 链到 object[property] 已有的回调上，避免覆盖其它扩展
function chainCallback(object, property, callback) {
    if (object == undefined) {
        return;
    }
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

// 重新计算节点高度 (参考 VHS 的 fitHeight)
// ComfyUI 的 DOM widget 高度由 widget.computeSize 决定，
// 改完 aspectRatio / hidden 后必须调一次让画布重排。
function fitHeight(node) {
    node.setSize([node.size[0], node.computeSize([node.size[0], node.size[1]])[1]]);
    node?.graph?.setDirtyCanvas(true, true);
}

// 节点类型注册时调用: 为该节点添加视频预览 widget + onExecuted 处理
function addVideoPreview(nodeType) {
    // 节点创建时: 建立 <video> 元素和 videopreview widget
    chainCallback(nodeType.prototype, "onNodeCreated", function () {
        const previewNode = this;
        const element = document.createElement("div");

        const previewWidget = this.addDOMWidget("videopreview", "preview", element, {
            serialize: false,
            hideOnZoom: false,
            getValue() { return element.value; },
            setValue(v) { element.value = v; },
        });

        // 根据视频宽高比计算 widget 高度，让画布上节点大小自适应
        // 返回 [width, height]；height<0 表示不占位 (未加载时)
        previewWidget.computeSize = function (width) {
            if (this.aspectRatio && !this.parentEl?.hidden) {
                let height = (previewNode.size[0] - 20) / this.aspectRatio + 10;
                if (!(height > 0)) {
                    height = 0;
                }
                this.computedHeight = height + 10;
                return [width, this.computedHeight];
            }
            return [width, -4]; // 没加载视频时不占高度
        };

        previewWidget.value = { hidden: false, paused: false, params: {}, muted: true };

        // 容器
        previewWidget.parentEl = document.createElement("div");
        previewWidget.parentEl.className = "topaz_preview";
        previewWidget.parentEl.style.width = "100%";
        element.appendChild(previewWidget.parentEl);

        // video 元素。性能优化要点:
        //   - controls=false: 原生控制条会和 ComfyUI 画布叠加渲染造成掉帧 (VHS 默认也是 false)
        //   - preload=auto: 让浏览器尽早缓冲，减少播放时的卡顿
        //   - playsinline: 移动端兼容
        //   - willChange 提示浏览器把 video 提升到独立合成层，避免和 litegraph 画布逐帧重绘合成
        previewWidget.videoEl = document.createElement("video");
        previewWidget.videoEl.controls = false;
        previewWidget.videoEl.loop = true;
        previewWidget.videoEl.muted = true;
        previewWidget.videoEl.preload = "auto";
        previewWidget.videoEl.setAttribute("playsinline", "");
        previewWidget.videoEl.setAttribute("disablepictureinpicture", "");
        previewWidget.videoEl.style.width = "100%";
        // 提升到独立图层: 这是解决节点内视频预览掉帧的关键。
        // ComfyUI 的 litegraph 画布在持续重绘，如果不隔离合成层，video 每帧都要
        // 和画布重新合成，高分辨率视频会明显掉帧。
        previewWidget.videoEl.style.willChange = "transform";
        previewWidget.videoEl.style.transform = "translateZ(0)";
        // 悬停时才显示控制条，避免常态渲染开销
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
            // 注意: loadedmetadata 回调里 this 不是 node，要用闭包里的 previewNode
            fitHeight(previewNode);
        });
        // 错误处理: 打印详情方便排查 (404/路径错误/编码不支持等)
        previewWidget.videoEl.addEventListener("error", (e) => {
            const code = previewWidget.videoEl.error?.code;
            const msg = previewWidget.videoEl.error?.message;
            console.warn(PREFIX, "video element error", "code=", code, "msg=", msg,
                "src=", previewWidget.videoEl.src);
            previewWidget.parentEl.hidden = true;
            fitHeight(previewNode);
        });
        previewWidget.parentEl.appendChild(previewWidget.videoEl);

        // 用 params (来自 gifs 数据) 设置 video.src
        previewWidget.updateSource = function () {
            const params = this.value?.params;
            if (!params?.filename) {
                return;
            }
            const q = Object.assign({}, params);
            q.timestamp = Date.now(); // 破坏缓存，确保每次刷新取最新文件
            this.parentEl.hidden = this.value.hidden ?? false;
            const fmt = String(q.format || "");
            console.log(PREFIX, "updateSource", "format=", fmt, "params=", params);
            // format 形如 "video/mp4": 视频分支用 <video>
            if (fmt.split("/")[0] === "video") {
                const url = api.apiURL("/view?" + new URLSearchParams(q));
                console.log(PREFIX, "video src =", url);
                this.videoEl.autoplay = !this.value.paused && !this.value.hidden;
                this.videoEl.src = url;
                this.videoEl.hidden = false;
                // 显式触发一次加载，某些情况下设 src 不会自动 load
                this.videoEl.load();
            } else {
                console.warn(PREFIX, "unexpected format, not video:", fmt);
                this.videoEl.hidden = true;
            }
        };
        previewWidget.callback = previewWidget.updateSource;
    });

    // 节点执行后: 接收后端返回的 ui.gifs，更新预览
    chainCallback(nodeType.prototype, "onExecuted", function (message) {
        if (message?.gifs?.length) {
            const previewWidget = this.widgets?.find((w) => w.name === "videopreview");
            if (previewWidget) {
                console.log(PREFIX, "onExecuted gifs =", message.gifs[0]);
                previewWidget.value.params = message.gifs[0];
                previewWidget.updateSource();
            } else {
                console.warn(PREFIX, "onExecuted: videopreview widget not found");
            }
        }
    });
}

app.registerExtension({
    name: "ComfyUI.TopazVideoAI",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData?.name === "TopazVideoAI") {
            addVideoPreview(nodeType);
        }
    },
});
