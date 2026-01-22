import torch
import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from modelscope.hub.snapshot_download import snapshot_download

# ========== 模型配置 ==========
MS_MODEL_ID = "qwen/Qwen2.5-Coder-1.5B-Instruct"
LOCAL_MODEL_DIR = "./models/Qwen2.5-Coder-1.5B-Instruct"
OUTPUT_DIR = "./output/luoguqwencoder-lora"

# ========== 下载模型 ==========
if not os.path.exists(LOCAL_MODEL_DIR):
    print(f"从ModelScope下载模型 {MS_MODEL_ID} 到 {LOCAL_MODEL_DIR}...")
    snapshot_download(
        repo_id=MS_MODEL_ID,
        local_dir=LOCAL_MODEL_DIR,
    )
    print("模型下载完成！")
else:
    print(f"本地已存在模型，直接加载：{LOCAL_MODEL_DIR}")

# ========== 加载 tokenizer ==========
tokenizer = AutoTokenizer.from_pretrained(
    LOCAL_MODEL_DIR,
    trust_remote_code=True,
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


# ========== 加载模型（4bit 量化）==========
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
model = AutoModelForCausalLM.from_pretrained(
    LOCAL_MODEL_DIR,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    # torch_dtype=torch.bfloat16,
    dtype=torch.bfloat16,
)
model.config.use_cache = False

# ========== LoRA 配置 ==========
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ========== 加载数据集 ==========
dataset = load_dataset("Misaka114514/luogu_dpo")

# ========== 数据预处理：转换为 ChatML 格式（TRL 0.27+ 标准）==========
def process_dataset_to_chatml(example):
    """
    将原始数据转换为 ChatML 格式
    - 所有格式化逻辑在此完成
    - trainer 只接收最终的 messages 字段
    - 符合 TRL 0.27+ 设计规范
    """
    try:
        # 1. 提取题目描述：从 "## 题目描述" 到 "【题目来源】"
        prompt = ""
        for conv in example["conversations"]:
            if conv["from"] == "human":
                conv_text = conv["value"].strip()
                start_marker = "## 题目描述"
                end_marker = "【题目来源】"
                start_idx = conv_text.find(start_marker)
                end_idx = conv_text.find(end_marker)
                if start_idx == -1:
                    start_idx = 0
                if end_idx == -1:
                    end_idx = len(conv_text)
                prompt = conv_text[start_idx:end_idx].strip()
                break

        # 2. 提取解答内容
        completion = example["chosen"]["value"].strip()

        # 3. 过滤无效样本
        if not prompt or not completion:
            return {"messages": [], "valid": False}

        # 4. 清洗题目文本
        prompt = prompt.replace("\n\n\n", "\n").strip()

        # 5. 构造用户指令（包含题目和要求）
        user_message = f"""你是一名信息学竞赛选手，请解决下面的问题。

【题目】
{prompt}

【要求】
- 将问题抽象成数学表述【较重要，但只需略微输出】
- 逐步分析合适算法与数据结构【重要，但只需略微输出】
- 给出完整的且易读性高的优质的C++代码【最重要，要完整输出】
- 将最终解决方案放在单个代码块中【重要】
- 请勿包含任何调试信息或额外输出
"""

        # 6. 转换为 ChatML 格式（TRL 0.27+ 标准）
        return {
            "messages": [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": completion}
            ],
            "valid": True
        }
    except (KeyError, Exception):
        return {"messages": [], "valid": False}

# 应用 ChatML 转换并过滤无效样本
dataset = dataset.map(process_dataset_to_chatml, remove_columns=dataset["train"].column_names)
dataset = dataset.filter(lambda x: x["valid"] is True)
print(f"过滤后有效样本数：训练集 {len(dataset['train'])} 条")

# ========== SFTConfig：仅包含训练参数（TRL 0.27+ 规范）==========
sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    num_train_epochs=2,
    learning_rate=1e-5,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    warmup_steps=100,
    fp16=False,
    bf16=True,
    logging_steps=50,
    save_steps=500,
    save_total_limit=2,
    report_to="none",
    gradient_checkpointing=True,
    remove_unused_columns=False,
    dataloader_pin_memory=False,
    neftune_noise_alpha=5.0,
)

# ========== SFTTrainer：只负责训练，不做格式化（TRL 0.27+ 规范）==========
trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset["train"],
    processing_class=tokenizer,
)

# ========== 训练 ==========
model.gradient_checkpointing_enable()
trainer.train()

# ========== 保存 ==========
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"训练完成，LoRA权重已保存到：{OUTPUT_DIR}")
