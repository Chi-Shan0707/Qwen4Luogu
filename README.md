# LuoguQwen — LoRA 微调示例

本仓库提供基于 Qwen 指令模型的 LoRA 微调示例和相关工具，包含训练脚本、评估脚本与示例数据。下文为项目说明、快速使用方法、基模型获取、推理方法、训练参数、数据来源及许可说明。


> 什么，你问我为什么要挑选Qwen2.5-1.5B-Instruct进行微调？<br>
- 那当然是因为它参数量小啦。<br>

> 什么，你继续问我为什么不挑选Qwen2.5-Coder-1.5B-Instruct进行微调？<br>
- ~~我如果在这阿里进行过代码训练上的模型进行微调，哪能看得出我微调的效果？~~<br>
~~好吧，其实是我问千问有什么参数量小的模型，它推荐了这个，然后我一时间忘记继续去搜集信息，直接开搞惹，结果训练到一半才在ModelScope上刷到Qwen2.5-Coder-1.5B-Instruct。PWP~~<br>
 ~~第一遍实在太差了，反正还要再训练一遍，还是弄Qwen2.5-Coder-1.5B-Instruct吧~~
 这个也太差劲了，上7B吧PwP<br>
 *不对，为什么疯狂报mismatch啊啊？从1.5B->7B我啥都没改啊？<br>
 *疯狂debug,疯狂研究格式*<br>
 算了，格式弄成所谓的标准型吧<br>
 7B根本跑不动啊，只能3B<br>
 ~~啊训练完了，参数根本上传不动啊？啊，huggingface也上传不动啊PwP~~

> 友情提示<br>
> UserWarning: Model has `tie_word_embeddings=True` and a tied layer is part of the adapter, but `ensure_weight_tying` is not set to True. This can lead to complications, for example when merging the adapter or converting your model to formats other than safetensors. Check the discussion here: https://github.com/huggingface/peft/issues/2777<br>
> *适配器和权重绑定的冲突，本质是PEFT 的 “独立层默认认知”与模型的 “权重绑定实际情况”的偏差，仅在合并适配器权重时暴露，ensure_weight_tying=True就是用来修正这个偏差的开关。而你现在看到的警告，就是 PEFT 在提醒你：“我现在按默认认知处理适配器，合并时可能会出偏差，记得打开纠偏开关～”。*

## 目录

- 项目概述
- 快速开始
- 下载基模型
- 使用 LoRA 权重进行推理
- 训练（SFT）关键参数
- 数据集来源
- 开源与许可证
- 联系方式与引用
- 依赖安装

---

## 项目概述

本仓库包含用于对 Qwen 指令模型在中文竞赛题数据上进行 LoRA 微调的示例：

- `train_sft.py`：基于 TRL 的 `SFTTrainer` 进行 LoRA 微调。
- `evaluate_model.py`：使用合并后的 LoRA 权重进行离线推理的示例。
- `download_dataset.py`：下载dataset
- `convert_dataset.py`：将原本的数据集格式转化为`SFTTrainer`可接受的`prompt`+`completion`格式
- `check_format.py`：检查转化后的格式

---

## 快速开始

**注意：本项目代码基于最新版本的库编写，请确保安装最新版本的依赖包以避免兼容性问题。**

1. 创建虚拟环境并安装依赖：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 或使用 Conda（推荐用于环境隔离）：
conda create -n luoguqwen python=3.10 -y
conda activate luoguqwen
pip install -r requirements.txt
```

2. 下载基模型（参见下文），设置 `LOCAL_MODEL_DIR` 与 `OUTPUT_DIR`，然后运行训练：

```bash
# 根据环境调整 accelerate 命令
accelerate launch train_sft.py
```

3. 训练完成后，运行推理示例：

```bash
python evaluate_model.py
```

---

## 下载基模型

推荐基模型：`qwen/Qwen2.5-1.5B-Instruct`（ModelScope）。获取方式示例：

- 从 ModelScope 下载（`train_sft.py` 使用 `snapshot_download`）：

```python
from modelscope.hub.snapshot_download import snapshot_download
snapshot_download(repo_id="qwen/Qwen2.5-Coder-1.5B-Instruct", local_dir="./models/Qwen2.5-Coder-1.5B-Instruct")
```

- 或通过 Hugging Face（如可用且许可允许）：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("qwen/Qwen2.5-Coder-1.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained("qwen/Qwen2.5-Coder-1.5B-Instruct")
```

注意：基模型体积较大并含有独立许可条款，请遵守相应使用条款。本仓库不包含完整基模型文件。

---

## 使用 LoRA 权重进行推理

训练完成后，`output/luoguqwen-lora/`（或 `train_sft.py` 中的 `OUTPUT_DIR`）将包含 LoRA 权重与 tokenizer 文件。

注意：本项目的 LoRA 权重也已经上传到 ModelScope（模型页路径）：
https://www.modelscope.cn/models/yuhanChi/Qwen4Luogu/tree/master/output/luoguqwencoder-lora 。
你可以直接从该页面下载所需文件并放到本地的 `output/luoguqwencoder-lora/` 目录，或在加载时使用该目录路径，之后按下述步骤加载和合并权重。

推荐流程：

1. 加载基模型与 tokenizer，使用 `peft` 加载 LoRA 权重，可选地合并权重以提升推理性能。

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("./models/Qwen2.5-Coder-1.5B-Instruct", device_map="auto", torch_dtype="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("./models/Qwen2.5-Coder-1.5B-Instruct", trust_remote_code=True)
model = PeftModel.from_pretrained(base, "./output/luoguqwencoder-lora")
model = model.merge_and_unload()

# 使用 model.generate 进行推理
```

2. `evaluate_model.py` 已包含最小推理示例，生成结果并保存至 `evaluation_results/evaluation_results.json`。

若仅公开 LoRA 权重，请在 README 中说明用户需自行下载基模型并提供路径。

---

## 训练（SFT）关键参数

**注意：本项目使用 TRL 0.27+ 的新特性（如 SFTConfig 和 ChatML 格式），请确保 TRL 版本 >= 0.27.0。**

`train_sft.py` 中使用的示例超参数：

- LoRA：`r=16`，`lora_alpha=32`，`lora_dropout=0.05`，目标模块：`q_proj, k_proj, v_proj, o_proj`。
- 训练：`num_train_epochs=2`，`per_device_train_batch_size=1`，`gradient_accumulation_steps=16`，`learning_rate=2e-4`。
- 可选：使用 4-bit (bnb NF4) 量化以节省显存。

上述参数为示例。请根据数据规模与显存调整超参数以获得稳定结果。

---

## 数据集来源

示例数据位于 `dataset_example/`，用于演示格式。`train_sft.py` 中使用`load_dataset("Misaka114514/luogu_dpo")` 

在线数据集链接（Hugging Face）：

- https://huggingface.co/datasets/Misaka114514/luogu_dpo

示例加载方法：

```python
from datasets import load_dataset
dataset = load_dataset("Misaka114514/luogu_dpo")
```

发布数据时请注意版权与隐私，只有在有权分享时才将数据包含在仓库内；否则提供数据获取或处理脚本并注明来源与许可。

---

## 开源与许可证

- 本仓库脚本示例采用 MIT 许可证（参见 `LICENSE`）。
- 基模型（`qwen/Qwen2.5-1.5B-Instruct`）为第三方提供，请遵守其原始许可证并自行获取基模型，本仓库不承担基模型的分发许可。
- 推荐公开的仓库内容：
  - `train_sft.py`、`evaluate_model.py`、`requirements.txt`、`dataset_example/`（若可分享）及 LoRA 权重文件夹 `output/luoguqwen-lora/`。
  - 不建议在仓库中托管完整基模型 `models/`，体积和许可均不适宜直接托管。

关于将 LoRA 权重上传至 GitHub 的说明：

- LoRA 权重通常为几十 MB 到几百 MB，可直接提交或使用 Git LFS 管理。若使用 Git LFS，请在 README 中注明并确保协作者安装 LFS。
- 本仓库 `.gitignore` 建议排除基模型目录 `models/` 与大型临时 checkpoint，同时允许 `output/luoguqwen-lora/` 被发布。

---

## 联系方式与引用

如使用本项目作为基线或在论文中引用，请在引用中保留作者信息与项目链接。

---

## 依赖安装

**重要：本项目代码基于最新版本的库编写，请务必使用最新版本以确保兼容性。**

```bash
# 推荐使用最新版本（必须 >= 指定的最低版本）
pip install -U \
  torch>=2.10.0 \
  transformers>=4.57.0 \
  datasets>=4.5.0 \
  accelerate>=1.12.0 \
  peft>=0.18.0 \
  trl>=0.27.0 \
  bitsandbytes>=0.49.0 \
  modelscope>=1.34.0

# 或使用 requirements.txt（推荐）
pip install -r requirements.txt
```

---

## English (translation)

This repository provides example scripts and tools for LoRA fine-tuning of a Qwen instruction model. The repository includes training and evaluation scripts and example data. The following sections describe the project, quick start, how to obtain the base model, inference with LoRA weights, key training parameters, dataset sources, licensing, and contact information.

Contents

- Project overview
- Quick start
- Download base model
- Inference with LoRA weights
- Training (SFT) key parameters
- Dataset source
- License and sharing
- Contact
- Dependencies

---

## Project overview

The repository contains examples for LoRA fine-tuning of the Qwen instruction model on Chinese contest problem data:

- `train_sft.py`: LoRA fine-tuning using TRL's `SFTTrainer`.
- `evaluate_model.py`: Example for offline inference using merged LoRA weights.
- `download_dataset.py`: Download dataset.
- `convert_dataset.py`: Convert the original dataset format to the `prompt`+`completion` format acceptable by `SFTTrainer`.
- `check_format.py`: Check the converted format.

---

## Quick start

**Note: This project code is written for the latest library versions. Please ensure you install the latest versions of dependencies to avoid compatibility issues.**

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Or use Conda (recommended for environment isolation):
conda create -n luoguqwen python=3.10 -y
conda activate luoguqwen
pip install -r requirements.txt
```

2. Download the base model (see below), set `LOCAL_MODEL_DIR` and `OUTPUT_DIR`, then run training:

```bash
# Adjust the accelerate command for your environment
accelerate launch train_sft.py
```

3. After training, run the inference example:

```bash
python evaluate_model.py
```

---

## Download base model

Recommended base: `qwen/Qwen2.5-1.5B-Instruct` (ModelScope). Example acquisition methods:

- From ModelScope (the training script uses `snapshot_download`):

```python
from modelscope.hub.snapshot_download import snapshot_download
snapshot_download(repo_id="qwen/Qwen2.5-1.5B-Instruct", local_dir="./models/Qwen2.5-1.5B-Instruct")
```

- Or via Hugging Face (if available and permitted):

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("qwen/Qwen2.5-1.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained("qwen/Qwen2.5-1.5B-Instruct")
```

Note: base models are large and carry separate license terms. Follow the provider's terms. This repository does not include the full base model files.

---

## Inference with LoRA weights

After training, `output/luoguqwen-lora/` (or `OUTPUT_DIR` in `train_sft.py`) will contain LoRA weights and tokenizer files.

Note: The LoRA weights for this project have also been uploaded to ModelScope at:
https://www.modelscope.cn/models/yuhanChi/Qwen4Luogu/tree/master/output/luoguqwencoder-lora .
You can download the files from that page and place them under the local `output/luoguqwencoder-lora/` directory, or point your loading code to that directory, then follow the steps below to load and optionally merge the adapter weights.

Recommended process:

1. Load the base model and tokenizer, use `peft` to load LoRA weights, optionally merge weights to improve inference performance.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("./models/Qwen2.5-1.5B-Instruct", device_map="auto", torch_dtype="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("./models/Qwen2.5-1.5B-Instruct", trust_remote_code=True)
model = PeftModel.from_pretrained(base, "./output/luoguqwen-lora")
model = model.merge_and_unload()

# Use model.generate for inference
```

2. `evaluate_model.py` includes a minimal inference example that generates results and saves them to `evaluation_results/evaluation_results.json`.

If only publishing LoRA weights, please explain in the README that users need to download the base model themselves and provide the path.

---

## Training (SFT) key parameters

**Note: This project uses TRL 0.27+ new features (such as SFTConfig and ChatML format), please ensure TRL version >= 0.27.0.**

Example hyperparameters used in `train_sft.py`:

- LoRA: `r=16`, `lora_alpha=32`, `lora_dropout=0.05`, target modules: `q_proj, k_proj, v_proj, o_proj`.
- Training: `num_train_epochs=2`, `per_device_train_batch_size=1`, `gradient_accumulation_steps=16`, `learning_rate=2e-4`.
- Optional: Use 4-bit (bnb NF4) quantization to save VRAM.

The above parameters are examples. Please adjust hyperparameters based on data size and VRAM to achieve stable results.

---

## Dataset source

Example data is located in `dataset_example/`, used to demonstrate format. 
Online dataset link (Hugging Face):

- https://huggingface.co/datasets/Misaka114514/luogu_dpo

Example loading method:

```python
from datasets import load_dataset
dataset = load_dataset("Misaka114514/luogu_dpo")
```

When publishing data, please be aware of copyright and privacy. Only include data in the repository if you have permission to share it; otherwise, provide data acquisition or processing scripts and note the source and license.

---

## License and sharing

- This repository's example scripts use the MIT license (see `LICENSE`).
- The base model (`qwen/Qwen2.5-1.5B-Instruct`) is provided by third parties. Please comply with its original license and obtain the base model yourself. This repository does not assume distribution rights for the base model.
- Recommended repository contents to publish:
  - `train_sft.py`, `evaluate_model.py`, `requirements.txt`, `dataset_example/` (if shareable), and LoRA weights folder `output/luoguqwen-lora/`.
  - It is not recommended to host the full base model `models/` in the repository, as both size and licensing make it unsuitable for direct hosting.

Notes on uploading LoRA weights to GitHub:

- LoRA weights are typically tens to hundreds of MB, can be committed directly or managed with Git LFS. If using Git LFS, note it in the README and ensure collaborators install LFS.
- This repository's `.gitignore` suggests excluding the base model directory `models/` and large temporary checkpoints, while allowing `output/luoguqwen-lora/` to be published.

---

## Contact

If using this project as a baseline or citing it in papers, please retain author information and project links in citations.

---

## Dependencies

**Important: This project code is written for the latest library versions. Please use the latest versions to ensure compatibility.**

```bash
# Recommended to use latest versions (must be >= specified minimum versions)
pip install -U \
  torch>=2.10.0 \
  transformers>=4.57.0 \
  datasets>=4.5.0 \
  accelerate>=1.12.0 \
  peft>=0.18.0 \
  trl>=0.27.0 \
  bitsandbytes>=0.49.0 \
  modelscope>=1.34.0

# Or use requirements.txt (recommended)
pip install -r requirements.txt
```