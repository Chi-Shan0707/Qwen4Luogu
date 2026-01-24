import torch
import os
from datasets import load_dataset, load_from_disk
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
# dataset = load_dataset("Misaka114514/luogu_dpo")
dataset = load_from_disk("./local_luogu_dataset")

def format_piece(sample):
    return sample["prompt"] + sample["completion"]

# ========== SFTConfig：仅包含训练参数（TRL 0.27+ 规范）==========
sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1, #一个gpu上同时处理1批数据
    gradient_accumulation_steps=16,# 每16步才更新一次参数
    # batch， 一次处理的样本
    # batchsize 16*1=16个
    num_train_epochs=2, # 完整遍历数据集2次
    learning_rate=1e-5, # 学习率，即参数更新的步长
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    warmup_steps=100,
    fp16=False,
    bf16=True,
    logging_steps=50,# 每50步记录一次日志
    save_steps=50, # 每500步保存一次模型检查点
    save_total_limit=10, #保存的检查点总数限制。设置为 2，只保留最新的 2 个检查点，防止磁盘空间占用过多。
    report_to="none",
    gradient_checkpointing=True, # 是否启用梯度检查点。设置为 True，通过重计算中间激活节省显存（以时间换空间），适合大模型训练。
    remove_unused_columns=False,
    dataloader_pin_memory=False,
    neftune_noise_alpha=5.0,# NEFTune 噪声参数。设置为 5.0，用于在训练中添加噪声以改善生成质量（减少生成重复）。值通常在 0-10 之间，0 表示禁用。
)

# ========== SFTTrainer：只负责训练，不做格式化（TRL 0.27+ 规范）==========
trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset["train"],
    processing_class=tokenizer,
    # formatting_func=format_piece,
)

# ========== 训练 ==========
model.gradient_checkpointing_enable()
trainer.train()

# ========== 保存 ==========
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"训练完成，LoRA权重已保存到：{OUTPUT_DIR}")
