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
from trl import SFTTrainer
# ========== 新增：导入ModelScope的模型下载工具 ==========
from modelscope.hub.snapshot_download import snapshot_download

# ========= 1. 模型配置（修改：新增ModelScope相关配置） =========
# ModelScope上Qwen2.5-1.5B-Instruct的官方repo ID
MS_MODEL_ID = "qwen/Qwen2.5-1.5B-Instruct"
# 本地模型保存路径（下载后从本地加载，避免重复下载）
LOCAL_MODEL_DIR = "./models/Qwen2.5-1.5B-Instruct"
# 训练输出路径（保持你的原有配置）
OUTPUT_DIR = "./output/luoguqwen-lora"

# ========== 新增：通过ModelScope下载模型到本地 ==========
# 检查本地是否已下载，未下载则从ModelScope下载（国内源速度快）
if not os.path.exists(LOCAL_MODEL_DIR):
    print(f"从ModelScope下载模型 {MS_MODEL_ID} 到 {LOCAL_MODEL_DIR}...")
    snapshot_download(
        repo_id=MS_MODEL_ID,
        local_dir=LOCAL_MODEL_DIR,
    )
    print("模型下载完成！")
else:
    print(f"本地已存在模型，直接加载：{LOCAL_MODEL_DIR}")

# ========= 2. 加载 tokenizer（修改：加载本地ModelScope下载的模型） =========
tokenizer = AutoTokenizer.from_pretrained(
    LOCAL_MODEL_DIR,  # 替换：从本地路径加载，而非Hugging Face远程ID
    trust_remote_code=True
)
# 补充：Qwen模型需要手动设置pad_token（避免训练时报错）
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # 右padding，提升训练稳定性

# ========= 3. 加载模型（4bit，省显存）（修改：加载本地模型） =========
# 新版 transformers 需要通过 BitsAndBytesConfig 定义量化参数
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
model = AutoModelForCausalLM.from_pretrained(
    LOCAL_MODEL_DIR,  # 替换：从本地路径加载，而非Hugging Face远程ID
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16,
)
# 补充：禁用模型梯度检查点外的不必要优化，节省显存

model.config.use_cache = False  # 训练时禁用cache，避免显存占用

# ========= 4. LoRA 配置（保留你的原有配置，适配Qwen2.5） =========
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]  # Qwen2.5的注意力层参数名，无需修改
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # 打印可训练参数（约0.1%，适配8GB显存）

# ========= 5. 加载数据集 + 数据预处理 =========
dataset = load_dataset("Misaka114514/luogu_dpo")

# 第一步：数据清洗 + 字段提取（核心修改）
def process_dataset(example):
    """
    从嵌套字段中提取有效内容，并做基础清洗
    """
    # 1. 提取human的题目描述（从conversations中取human的value）
    try:
        # 遍历conversations，找到from="human"的value（题目内容）
        prompt = ""
        for conv in example["conversations"]:
            if conv["from"] == "human":
                # 只保留从「## 题目描述」开始，到【题目来源】前的部分
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
        
        # 2. 提取gpt的解答内容（从chosen的value中取）
        chosen = example["chosen"]["value"].strip()
        
        # 3. 清洗：过滤空值/无效数据
        if not prompt or not chosen:
            return {"prompt": "", "chosen": "", "valid": False}
        
        # 4. 清洗：去除多余的换行/空格（保留代码块格式）
        prompt = prompt.replace("\n\n\n", "\n").strip()
        
        return {
            "prompt": prompt,
            "chosen": chosen,
            "valid": True  # 标记为有效数据
        }
    except KeyError:
        # 字段缺失的样本标记为无效
        return {"prompt": "", "chosen": "", "valid": False}

# 应用数据处理函数
dataset = dataset.map(process_dataset)
# 过滤无效样本（valid=False的全部删掉）
dataset = dataset.filter(lambda x: x["valid"] is True)
print(f"过滤后有效样本数：训练集 {len(dataset['train'])} 条")



# ========= 6. 数据格式化（非常重要，保留你的原有逻辑） =========
def format_example(example):
    """
    把 luogu_dpo 的 chosen 变成 SFT 指令
    """
    return f"""你是一名信息学竞赛选手，请解决下面的问题。

【题目】
{example["prompt"]}

【要求】
- 给出清晰的算法思路
- 分析时间复杂度
- 给出可通过的C++
- 请勿包含任何调试信息或额外输出。
- 将最终解决方案放在单个代码块中

【解答】
{example["chosen"]}
"""

# ========= 7. 训练参数（为 8GB 显存精调，保留你的原有配置） =========
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,      # 8GB显存必选1
    gradient_accumulation_steps=16,     # 等效batch=16，平衡训练效果
    num_train_epochs=2,                 # 2轮避免过拟合
    learning_rate=2e-4,                 # Qwen2.5小模型适配该学习率
    fp16=True,                          # 混合精度，节省显存
    logging_steps=50,
    save_steps=500,
    save_total_limit=2,                 # 只保留最新2个检查点，节省磁盘
    report_to="none",                   # 不使用wandb，本地训练
    # 补充：新增显存优化参数（适配8GB显存）
    gradient_checkpointing=True,

# 或在外部用
# model.gradient_checkpointing_enable()
# 代替
    remove_unused_columns=False,
    dataloader_pin_memory=False         # 关闭pin_memory，减少显存占用
)

# ========= 8. Trainer（保留你的原有配置，补充关键参数） =========
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,          # TRL >=0.7 uses processing_class instead of tokenizer
    train_dataset=dataset["train"],
    formatting_func=format_example,
    max_seq_length=2048,                # 适配竞赛题+代码的长度
    args=training_args,
    # 补充：SFTTrainer专属优化，避免样本截断
    packing=False,                      # 关闭packing，适配长文本
    neftune_noise_alpha=5.0             # 轻微噪声，提升泛化能力
)

# ========= 9. 开始训练 =========
trainer.train()

# ========= 10. 保存 LoRA =========
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"训练完成，LoRA权重已保存到：{OUTPUT_DIR}")