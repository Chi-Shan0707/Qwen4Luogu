import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json
import os

# 模型路径
BASE_MODEL_DIR = "./models/Qwen2.5-Coder-1.5B-Instruct"
LORA_DIR = "./output/luoguqwencoder-lora"
OUTPUT_DIR = "./evaluation_results"

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# 加载基模型
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_DIR,
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

# 加载LoRA权重
model = PeftModel.from_pretrained(base_model, LORA_DIR)
model = model.merge_and_unload()  # 合并权重以提高推理速度


print("model max pos:", model.config.max_position_embeddings)
print("tokenizer max len:", tokenizer.model_max_length)

# 测试题目（可以从dataset_example中取几个）
test_problems = [
    {
        "title": "P8866 喵了个喵",
        "description": """小 E 喜欢上了一款叫做《喵了个喵》的游戏。这个游戏有一个牌堆和 n 个可以从栈底删除元素的栈，任务是要通过游戏规则将所有的卡牌消去。开始时牌堆中有 m 张卡牌，从上到下的图案分别是 a_1,a_2,…,a_m
​所有的卡牌一共有 k 种图案，从 1 到 k 编号。牌堆中每一种图案的卡牌都有偶数张。开始时所有的栈都是空的。这个游戏有两种操作：

- 1. 选择一个栈，将牌堆顶上的卡牌放入栈的顶部。如果这么操作后，这个栈最上方的两张牌有相同的图案，则会自动将这两张牌消去。
- 2. 选择两个不同的栈，如果这两个栈栈底的卡牌有相同的图案，则可以将这两张牌消去，原来在栈底上方的卡牌会成为新的栈底。如果不同，则什么也不会做。

这个游戏一共有 T 关，小 E 一直无法通关。请你帮小 E 设计一下游戏方案，即对于游戏的每一关，给出相应的操作序列使得小 E 可以把所有的卡牌消去。
输入格式：
第一行包含一个正整数 T，表示数据组数。

接下来一共 T 组数据，在每组数据中：

第一行包含三个正整数 n,m,k，分别表示栈的个数、卡牌的个数、卡牌上图案的种类。

第二行包含 m 个正整数，分别表示 a_1 ,a_2,…,a_m，分别从上到下表示牌堆中卡牌的图案。

输入数据保证有解。

输出格式：
对于每一组数据，输出若干行。

其中第一行包含一个正整数 op，表示操作的次数。你需要保证 m≤op≤2×m。

接下来 op 行，每行包含两个或三个正整数，整数之间用一个空格隔开。

若为两个整数 1 s，则进行一次第一个操作并选择栈 s。

若为三个整数 2 s1 s2，则进行一次第二个操作并选择栈 s1 和 s2。

你需要保证 1≤s,s1,s2≤n，且 s1!=s2。

数据范围：
对于 30% 的数据，满足 2 ≤ m,n ≤ 10。
对于 100% 的数据，满足 2 ≤ m,n ≤ 50。

样例输入：
1
2 4 2
1 2 1 2

样例输出：
5
1 1
1 1
1 2
2 1 2
1 1"""
    }
]

def generate_response(problem):
    user_message = f"""你是一名信息学竞赛选手，请解决下面的问题。

【题目】
{problem['description']}

【要求】
- 将问题抽象成数学表述【较重要，略微输出】
- 逐步思考合适算法【略微输出】
- 给出完整的且易读性高的优质的C++代码【最重要，要完整输出】
- 将最终解决方案放在单个代码块中【重要】
- 请勿包含任何调试信息或额外输出【你不能输出】

"""

    messages = [{"role": "user", "content": user_message}]

    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    )
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=3072,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
    return response

# 评估
results = []
for i, problem in enumerate(test_problems):
    print(f"正在评估题目 {i+1}: {problem['title']}")
    response = generate_response(problem)
    print("模型响应:")
    print(response)
    print("-" * 50)

    # 保存结果
    results.append({
        "problem": problem["title"],
        "response": response
    })

# 保存到文件
os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(f"{OUTPUT_DIR}/evaluation_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"评估完成，结果已保存到 {OUTPUT_DIR}/evaluation_results.json")
