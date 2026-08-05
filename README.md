## Language

- [English](#english)
- [中文](#中文)

### English
**Requirements:**

Licensed installation of Topaz Video AI (product renamed to **Topaz Video** in 2025; both old "Topaz Video AI" and new "Topaz Video" installations are auto-detected)

**Installation:**

method 1: search 'TopazVideoAI' in manager and install it.

method 2: git clone this project to custom_nodes folder
```
git clone https://github.com/sh570655308/ComfyUI-TopazVideoAI.git
```

**Setup (updated for Topaz Video / ffmpeg 8.1):**

The node auto-detects the Topaz install directory and model directory on startup, so manual environment variables are usually NOT required anymore. If you upgraded from the old "Topaz Video AI" version, you may want to **delete the stale `TVAI_MODEL_DIR` / `TVAI_MODEL_DATA_DIR` user environment variables** — after the rename they can point at the old install's Python runtime folder (`...\Topaz Video AI\models`) which no longer contains model files, causing "Model not found" errors. The plugin detects this and falls back to the new path (`C:\ProgramData\Topaz Labs LLC\Topaz Video\models`), but removing the stale variables avoids confusion.

If auto-detection ever fails, set them manually:
- `TVAI_MODEL_DIR` = `C:\ProgramData\Topaz Labs LLC\Topaz Video\models`
- `TVAI_MODEL_DATA_DIR` = (same value)

![image](https://github.com/user-attachments/assets/996eba42-a356-4324-a697-706536cb4da4)
The path to ffmpeg is specified within the node, defaulting to the auto-detected install dir (`C:\Program Files\Topaz Labs LLC\Topaz Video` for the new version). This specific ffmpeg is mandatorily designated only for the processes of upscaling and frame interpolation. Users have the flexibility to customize ffmpeg in their environment variables for handling other tasks.

Close the GUI, open shell terminal and log in:

```cd "C:\Program Files\Topaz Labs LLC\Topaz Video"```

```.\login```

The path may vary when you have a custom installation path. Then close the shell terminal, now you can use this node normally.

**Usage:**

Simply connect this node between video output and video save.
Workflows contained in examples folder.
The model dropdowns are populated dynamically by scanning the installed model directory, so any new models Topaz ships will appear automatically (no code change needed).

**Upscale parameters (new — replaces the old single `blend` slider):**

Topaz's ffmpeg 8.1 removed the old `blend` parameter and split it into fine-grained controls. Each slider maps 1:1 to the `tvai_up` filter option of the same name (see `ffmpeg -h filter=tvai_up`):

| Parameter | Range | Meaning |
|---|---|---|
| `compression` | -1 ~ 1 | Remove compression artifacts (blockiness, mosquito noise). Default 1.0 |
| `noise` | -1 ~ 1 | Remove ISO noise; negative favors keeping detail |
| `details` | -1 ~ 1 | Recover fine texture lost to in-camera denoising |
| `halo` | -1 ~ 1 | Suppress ringing/halo from oversharpening |
| `blur` | -1 ~ 1 | Extra sharpening; raise if the input looks soft |
| `preblur` | -1 ~ 1 | Pre-blur handling; negative leans toward anti-aliasing / moiré |

Old workflows that still carry a `blend` value are handled gracefully — `blend` is ignored and missing new parameters fall back to their defaults, so existing pipelines keep working.

**Notification:**

The default upscale model is **Proteus (`prob-4`)** — the most general-purpose model, suitable for a wide range of source material. You can of course pick any other model from the dropdown.

The models have scaling limitations; for example, thm-2 is fixed at a 1x scale. As not all models have undergone comprehensive testing and standardization for various scaling factors, it is advisable to use either 2x or 4x. If you encounter errors when attempting a 4x scale, please default to using a 2x scale.
This node is designed for short AI generated videos. I didn't test it with long video, because comfyui transfers video as image batch; the node encodes and decodes, which costs longer time than the Topaz Video GUI.

**Why Starlight / Astra models are NOT in the list:**

Topaz's Starlight family (Starlight Mini/Fast/HQ/Sharp/Precise, exposed as the `astra` / `astrafast` / `astrahq` / `astrasharp` / `slp-*` models) is built on the new **Neuroserver / diffusion** architecture. These models are **not** invocable through the `tvai_up` / `tvai_fi` ffmpeg filters that this node relies on — the GUI runs them via a separate `neuroserver` runtime / `runner` pipeline, and Topaz has stated it is phasing out CLI support. Any attempt to call them via `tvai_up=model=astra...` fails immediately. They are therefore deliberately excluded from the model dropdowns so that every listed model actually works. If you need Starlight, render in the Topaz Video GUI directly.

Common errors:
- `No such filter: 'tvai_up'` — the ffmpeg path is wrong; make sure it points at Topaz's bundled ffmpeg.
- `Model not found: <name>` — the model directory resolves to a stale path (typical after upgrading from the old "Topaz Video AI"). Delete the `TVAI_MODEL_DIR` / `TVAI_MODEL_DATA_DIR` environment variables and restart ComfyUI so the plugin auto-detects the new `...\Topaz Video\models` directory.

### 中文
**要求：**

已安装的 Topaz Video AI，要登录账户。（2025 年 Topaz 把产品改名为了 **Topaz Video**，本节点会自动检测新旧两种安装，无需手动处理）

**安装：**

方法1.通过comfyui manager搜索topazvideoai后安装

方法2.将此项目git clone到 custom_nodes 文件夹

```
git clone https://github.com/sh570655308/ComfyUI-TopazVideoAI.git
```

**配置（适配新版 Topaz Video / ffmpeg 8.1）：**

节点启动时会自动探测 Topaz 安装目录和模型目录，通常**不再需要**手动配置环境变量。如果你是从旧版 "Topaz Video AI" 升级上来的，建议**删除残留的 `TVAI_MODEL_DIR` / `TVAI_MODEL_DATA_DIR` 用户环境变量** —— 改名后它们可能指向旧安装的 Python 运行目录（`...\Topaz Video AI\models`），那里已经没有模型文件，会报 "Model not found"。本节点会探测到这种情况并自动回退到新路径（`C:\ProgramData\Topaz Labs LLC\Topaz Video\models`），但删掉残留变量能避免困惑。

如果自动探测失败，可手动设置：
- `TVAI_MODEL_DIR` = `C:\ProgramData\Topaz Labs LLC\Topaz Video\models`
- `TVAI_MODEL_DATA_DIR` = （同上）

![image](https://github.com/user-attachments/assets/996eba42-a356-4324-a697-706536cb4da4)

ffmpeg 路径在节点中指定，默认值是自动探测到的安装目录（新版为 `C:\Program Files\Topaz Labs LLC\Topaz Video`）。此 ffmpeg 只在放大和补帧过程强制指定，可以自行在环境变量中自定义 ffmpeg 负责其他环节。

设置完成后关闭 GUI，打开 shell 终端输入：

```cd "C:\Program Files\Topaz Labs LLC\Topaz Video"```

```.\login```

之后就可以正常使用了，

**使用：**

接在视频输入前即可。

examples 文件夹中包含工作流。
模型下拉列表是启动时动态扫描模型目录生成的，Topaz 以后发布的新模型会自动出现，无需改代码。

**放大参数（新版 —— 取代了旧的单一 `blend` 滑块）：**

Topaz 的 ffmpeg 8.1 移除了旧的 `blend` 参数，拆分成了更精细的控制项。每个滑块都一一对应 `tvai_up` 滤镜的同名选项（详见 `ffmpeg -h filter=tvai_up`）：

| 参数 | 范围 | 含义 |
|---|---|---|
| `compression` | -1 ~ 1 | 去除压缩块效应/蚊噪，默认 1.0 |
| `noise` | -1 ~ 1 | 去除 ISO 噪点，负值偏重保留细节 |
| `details` | -1 ~ 1 | 恢复相机降噪丢失的细节纹理 |
| `halo` | -1 ~ 1 | 抑制过锐产生的振铃/光晕 |
| `blur` | -1 ~ 1 | 额外锐化，输入偏软时调高 |
| `preblur` | -1 ~ 1 | 预处理模糊，负值偏重抗锯齿/摩尔纹 |

旧工作流里残留的 `blend` 值会被安全忽略，缺失的新参数自动补默认值，因此已有工作流无需重连也能继续运行。

**注意事项：**

默认放大模型是 **Proteus（`prob-4`）** —— 通用性最强，适合各种素材。你当然也可以在下拉框里选其它模型。

模型有倍数限制，例如 thm-2 只能 1 倍，由于未对全部模型的倍数进行测试和规范化，所以放大倍数请使用 2 或者 4，当 4 报错时请使用 2。
此节点专为 AI 生成的短视频设计。由于 ComfyUI 以图像批次方式传输视频，节点需要进行编码和解码，因此相比 Topaz Video 图形界面处理时间更长，故未对长视频进行测试。

**为什么 Starlight / Astra 系列模型不在列表里：**

Topaz 的 Starlight 系列（Starlight Mini/Fast/HQ/Sharp/Precise，对应 `astra` / `astrafast` / `astrahq` / `astrasharp` / `slp-*` 等模型）基于全新的 **Neuroserver / 扩散模型** 架构。这些模型**无法**通过本节点依赖的 `tvai_up` / `tvai_fi` ffmpeg 滤镜调用 —— GUI 是通过独立的 `neuroserver` 运行时 / `runner` 流程来跑它们的，而且 Topaz 已宣布逐步停止 CLI 支持。任何 `tvai_up=model=astra...` 的调用都会立即失败。因此这些模型被有意排除在 dropdown 之外，保证列表里每一个模型都能真正跑通。如果你需要 Starlight，请直接在 Topaz Video GUI 里渲染。

常见错误：
- `No such filter: 'tvai_up'` —— ffmpeg 路径不对，确保指向 Topaz 自带的 ffmpeg。
- `Model not found: <名称>` —— 模型目录解析到了残留的旧路径（通常是从旧版 "Topaz Video AI" 升级导致）。删除 `TVAI_MODEL_DIR` / `TVAI_MODEL_DATA_DIR` 环境变量并重启 ComfyUI，让插件自动探测新的 `...\Topaz Video\models` 目录。
