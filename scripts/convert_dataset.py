from datasets import load_from_disk

# 加载整个数据集（DatasetDict，包含train等splits）
dataset = load_from_disk("./local_luogu_dpo")


dataset = dataset['train']


def convert (example) :
    try :
        prompt = """你是一个C++代码生成器，专为信息学竞赛普及组难度的题目设计。请严格遵守：
1. 仅输出C++代码，不包含任何其他文本。
2. 代码必须正确、无编译错误、逻辑正确。
3. 代码应简洁、朴实，避免冗余。
4. 在代码中，对于关键步骤，用注释简要说明数学本质（如：// 数学本质：求最大公约数）。
5. 不输出任何思考过程或解释。
题目内容如下：
"""

        input = ""
        for conv in example.get("conversations",[]) :
            human_input = conv.get("value","").strip()
            start_marker = "题目描述"
            end_marker = "样例"
            start_index = human_input.find(start_marker)
            end_index = human_input.find(end_marker)
            if start_index == -1 :
                return {
                    "text":"",
                    "valid":False
                }
            if end_index == -1 :
                return {
                    "text":"",
                    "valid":False
                    }
                
            input += human_input[start_index:end_index].strip()
            break

        prompt = prompt + input

        completion = ""

        choice=example.get("chosen",{}) 
        gpt_output = choice.get("value","").strip()
        start_marker = "#include"
        end_marker = "```\n"
        start_index = gpt_output.find(start_marker)
        end_index = gpt_output.find(end_marker)
        if start_index == -1 :
            return {"text":"",
                    
                    "valid":False
                    }
        if end_index == -1 :
            end_index = len(gpt_output)

        completion += gpt_output[start_index:end_index].strip().replace('```', '')
        
        # Format as ChatML
        text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{completion}<|im_end|>"
        
        return {
             "text": text,
            "valid": True
         }
    
    except (KeyError,Exception) as e :
        #except (KeyError, Exception) as e：捕获两种异常类型：
# KeyError：字典中访问不存在的键时抛出（例如，example["conversations"]不存在）。
# Exception：所有其他异常的基类（包括AttributeError、ValueError等）。
# 将异常对象赋值给变量e（可选，用于调试，但代码中未使用）。
        return {
            "text": "",
            "valid": False
        }

########
#
#默认（非批处理，batched=False）
#example 是一个字典（Python dict），键是列名，值是该样本对应的标量（例如 example["conversations"] 是该单个样本的 conversations 列的值，通常是一个 list/object）。
#批量模式（batched=True）
#example 是一个字典，键是列名，值是“列表”（批次中每个样本对应一项）。例如 example["conversations"] 会是一个列表，长度等于 batch 大小。
#
########
# 使用更稳健的处理流程：map -> filter -> remove temp cols -> 保存为 DatasetDict
# 新的、稳健的提取函数（不替换原有函数，方便回滚）

mapped = dataset.map(convert, batched=False, remove_columns=dataset.column_names)
print('After map, columns:', mapped.column_names, 'len=', len(mapped))
# 过滤有效样本
mapped = mapped.filter(lambda x: x['valid'])
print('After filter, len=', len(mapped))
# 去掉 valid 列，保留 prompt 和 completion 列
if 'valid' in mapped.column_names:
    mapped = mapped.remove_columns(['valid'])

from datasets import DatasetDict
out = DatasetDict({'train': mapped})
out.save_to_disk('./local_luogu_dataset')
print('Saved converted dataset to ./local_luogu_dataset with train len=', len(out['train']))


