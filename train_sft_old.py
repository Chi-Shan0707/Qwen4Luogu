import torch
import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from modelscope.hub.snapshot_download import snapshot_download

MS_MODEL_ID = "qwen/Qwen2.5-1.5B-Instruct"
LOCAL_MODEL_DIR = "./models/Qwen2.5-1.5B-Instruct"
OUTPUT_DIR = "./output/luoguqwen-lora"

if not os.path.exists(LOCAL_MODEL_DIR):
    print(f"从ModelScope下载模型 {MS_MODEL_ID} 到 {LOCAL_MODEL_DIR}...")
    snapshot_download(
        repo_id=MS_MODEL_ID,
        local_dir=LOCAL_MODEL_DIR,
    )
    print("模型下载完成！")
else:
    print(f"本地已存在模型，直接加载：{LOCAL_MODEL_DIR}")

tokenizer = AutoTokenizer.from_pretrained(
    LOCAL_MODEL_DIR,
    trust_remote_code=True,
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

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
    dtype=torch.float16,
)
model.config.use_cache = False

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

dataset = load_dataset("Misaka114514/luogu_dpo")

def process_dataset(example):
    """
    保留从“## 题目描述”到“【题目来源】”的正文，prompt 为纯题目描述
    """
    try:
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

        completion = example["chosen"]["value"].strip()

        if not prompt or not completion:
            return {"prompt": "", "completion": "", "valid": False}

        prompt = prompt.replace("\n\n\n", "\n").strip()

        return {
            "prompt": prompt,
            "completion": completion,
            "valid": True,
        }
    except KeyError:
        return {"prompt": "", "completion": "", "valid": False}

dataset = dataset.map(process_dataset)
dataset = dataset.filter(lambda x: x["valid"] is True)
print(f"过滤后有效样本数：训练集 {len(dataset['train'])} 条")

def format_example(example):
    """
    格式化为比赛风格的SFT指令，返回字典以匹配 TRL 的预期字段
    """
    instruction = f"""你是一名信息学竞赛选手，请解决下面的问题。\n\n【题目】\n{example['prompt']}\n\n【要求】\n- 给出清晰的算法思路\n- 分析时间复杂度\n- 给出可通过的C++\n- 请勿包含任何调试信息或额外输出。\n- 将最终解决方案放在单个代码块中\n\n"""
    # completion 保留为原始答案文本（不包含额外格式），由 TRL 在后续步骤加入 EOS 等处理
    return {
        "prompt": instruction,
        "completion": example["completion"]
    }


sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,

    # === 文本 / tokenizer 相关 ===
    max_length=2048,
    # 为了兼容提供的 formatting_func，关闭 completion_only_loss（否则二者冲突）
    completion_only_loss=False,
    packing=False,
    

    # === 训练超参数（原 TrainingArguments 内容）===
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    num_train_epochs=2,
    learning_rate=2e-4,
    fp16=True,

    logging_steps=50,
    save_steps=500,
    save_total_limit=2,
    report_to="none",

    gradient_checkpointing=True,
    remove_unused_columns=False,
    dataloader_pin_memory=False,

    # === 可选：正则化 ===
    neftune_noise_alpha=5.0,
)
trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset["train"],
    processing_class=tokenizer,   # 指定 tokenizer/processing 对象以便 TRL 做 tokenize
    formatting_func=format_example
)

model.gradient_checkpointing_enable()
model.config.use_cache = False
trainer.train()
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"训练完成，LoRA权重已保存到：{OUTPUT_DIR}")