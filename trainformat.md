```python
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

```