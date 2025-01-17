# ComfyUI-FitDiT

本项目基于https://github.com/BoyuanJiang/FitDiT，只是将其包装为ComfyUI的节点。

## Installation

第一步 申请访问 FitDiT 的[模型权重](https://huggingface.co/BoyuanJiang/FitDiT)，然后将模型克隆到*local_model_dir*（例如：_models/FitDiT_）目录

第二步 将本项目克隆到*ComfyUI*的*custom_nodes*目录下

第三步 在 ComfyUI 中启用 FitDiT 节点
示例工作流: [example_workflow/fitdit.json](example_workflow/fitdit.json)

> **提示：**第一次运行会下载 _openai/clip-vit-large-patch14_ 和 _laion/CLIP-ViT-bigG-14-laion2B-39B-b160k_ 模型，请耐心等待（如果你本地之前没有的话，需要科学）。

## TODO LIST

- 自定义遮罩优化
