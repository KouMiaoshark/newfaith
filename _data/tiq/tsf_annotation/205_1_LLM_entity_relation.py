import os
import json
from openai import OpenAI

client = OpenAI(
    api_key="sk-963eb925b9954830a0b817307e0a449b",
    base_url="https://api.deepseek.com"
)

INPUT_FILE = "simplified_317test.json"
OUTPUT_FILE = "历史提示词处理/solution9_simplified_317test.json"

BATCH_SIZE = 20

start_index = 0
MAX_N = 1  # 跑到 MAX_N 条；如果数据不足会自动停止

def strip_code_fences(text: str) -> str:
    """去掉 ```json ... ``` 或 ``` ... ``` 包裹"""
    t = text.strip()
    if t.startswith("```"):
        first_nl = t.find("\n")
        if first_nl != -1:
            t = t[first_nl + 1:]
        t = t.strip()
        if t.endswith("```"):
            t = t[:-3].strip()
    return t.strip()

def parse_any_json_objects(text: str):
    """
    尽可能把模型输出解析成 [{Id:..., Relation/relation:...}, ...]
    兼容：
    - 标准 JSON 数组
    - 单个 JSON 对象
    - 连续多个 JSON 对象（中间只有空白/换行/逗号/方括号）
    - 多行缩进的 JSON 对象（raw_decode 可处理）
    """
    t = strip_code_fences(text)

    # 1) 尝试直接 loads（数组/对象）
    try:
        obj = json.loads(t)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            return [obj]
    except Exception:
        pass

    # 2) 连续对象解析（raw_decode 循环）
    decoder = json.JSONDecoder()
    i = 0
    objs = []
    while i < len(t):
        # 跳过空白/逗号/方括号
        while i < len(t) and t[i] in " \r\n\t,[]":
            i += 1
        if i >= len(t):
            break
        try:
            obj, end = decoder.raw_decode(t, i)
            objs.append(obj)
            i = end
        except json.JSONDecodeError:
            break
    return objs

# 读取输入数据
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# 读取已存在输出（用于断点续跑 + 去重）
if os.path.exists(OUTPUT_FILE):
    try:
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            existing = json.load(f)
        if not isinstance(existing, list):
            existing = []
    except Exception:
        existing = []
else:
    existing = []

# 用 Id 做索引，避免重复写入
existing_by_id = {item.get("Id"): item for item in existing if isinstance(item, dict) and "Id" in item}

if existing_by_id:
    next_id = max(existing_by_id.keys()) + 1
else:
    next_id = 1
limit = min(MAX_N, len(data))

# start_index = limit - question_num  # start_index计算方式已更正

# 批次处理
for start in range(start_index, limit, BATCH_SIZE):
    batch = data[start:start + BATCH_SIZE]
    if not batch:
        break

    instruction_text = """请输出 JSON（不要 markdown 代码块），每条为： 
            {"Id": <int>, "Relation": <string>, "entity": [<string>, <string>, ...]}

            任务：给定一个自然语言问题，从中提取核心的实体、关系，并根据问题的语境和背景信息推理出可能未显性提到的实体。生成的TSF应能补充必要的背景信息，如历史事件、时间范围和其他对答案检索至关重要的实体信息。

            硬性规则（必须遵守）：

            **entity提取规则：**
            1. 请从问题中提取出所有显性和隐性实体。实体可以是人物、地点、事件、组织等。
            2. 需要注意从问题中推理出可能没有显式提到但对于答案有帮助的实体（如历史事件、背景等）。
            3. 对于涉及时间的问题（如“在2016年”，或“在东京奥运会期间”），需要将时间相关的信息也作为实体的一部分。

            **relation提取规则：**
            4. 请从问题中提取出主体关系，确保问题中的关系能够精准描述问题的核心。例如：提取“谁是...的领导人”中的“领导人”作为关系。
            5. 如果问题中包含时间范围或事件，关系需要能够结合时间推理。举个例子，“2016年到2020年间”是时间范围，关系应与这个时间段相关联。
            6. 对于隐性时间问题，提取出隐藏在问题背景中的时间信息，并适当推理出对应的关系。

             输出示例：  
examples: [
  {
    "input": {
      "Id": 6005,
      "Question": "Which international football team did Roy Keane join after being a member of the Republic of Ireland national under-21 football team?"
    },
    "output": {
      "Id": 6005,
      "Relation": "joined international football team after being a member of Republic of Ireland national under-21 football team",
      "entity": ["Roy Keane", "Republic of Ireland national under-21 football team", "Republic of Ireland national football team", "Irish football", "international football"]
    }
  },

  {
    "input": {
      "Id": 6017,
      "Question": "During Ashley Cooper's tennis career, in which Grand Slam events did he win his three singles titles?"
    },
    "output": {
      "Id": 6017,
      "Relation": "won three singles titles in Grand Slam events during tennis career",
      "entity": ["Ashley Cooper", "tennis career", "Grand Slam events", "singles titles", "tennis history", "Australian Open", "Wimbledon", "US Open"]
    }
  },

  {
    "input": {
      "Id": 6071,
      "Question": "What ethnic group did Milo O'Shea belong to before he passed away in Manhattan, New York at the age of 86?"
    },
    "output": {
      "Id": 6071,
      "Relation": "belonged to ethnic group before death in Manhattan, New York",
      "entity": ["Milo O'Shea", "ethnic group", "Manhattan", "New York", "death", "Irish ethnicity", "Irish diaspora"]
    }
  }
  
]

    """

    # ChatGPT API interaction code and batch processing as before

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": instruction_text},
    ]

    for item in batch:
        messages.append({"role": "user", "content": json.dumps({"Id": item["Id"], "Question": item["Question"]})})

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        stream=False
    )

    raw = response.choices[0].message.content
    objs = parse_any_json_objects(raw)

    # 建立 Id -> Relation 映射（兼容 Relation/relation 两种键）
    id2rel = {}
    for obj in objs:
        if not isinstance(obj, dict):
            continue
        _id = obj.get("Id")
        rel = obj.get("Relation", obj.get("relation"))
        entities = obj.get("entity", [])
        if _id is not None and isinstance(rel, str):
            id2rel[int(_id)] = {"Relation": rel.strip(), "entity": entities}

    # 写回 batch，并更新 existing_by_id
    for item in batch:
        _id = int(item["Id"])
        if _id in id2rel:
            item.update(id2rel[_id])  # 更新 Relation 和 entity 字段
        # 合并到总结果（覆盖更新）
        existing_by_id[_id] = item

    # 每 10 条保存一次（同一个文件）
    merged = [existing_by_id[k] for k in sorted(existing_by_id.keys())]
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"[saved] {start}~{start+len(batch)-1}  (total saved: {len(merged)})")

print(f"Done. Output -> {OUTPUT_FILE}")