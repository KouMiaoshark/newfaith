import os
import json
from openai import OpenAI

client = OpenAI(
    api_key="sk-963eb925b9954830a0b817307e0a449b",
    base_url="https://api.deepseek.com"
)

INPUT_FILE = "simplified_train.json"
OUTPUT_FILE = ("simplify_rela_train_solution5_TempAns_16723-16823.json")

# INPUT_FILE = "simplified_dev.json"
# OUTPUT_FILE = "simplify_rela_dev_solution1.json"
BATCH_SIZE = 10

#总共7723+6000=13723， 所以要计算第167
question_num=120
MAX_N = 12823  # 跑到 MAX_N 条；如果数据不足会自动停止

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
# limit = len(data)#这是全跑

#懂了 start_index不可以是15723，因为数据总共是 6000+7723
start_index = limit-question_num
#这样就可以，跑MAX_N 前question_num个问题



#len(data)
for start in range(start_index, limit, BATCH_SIZE):
    batch = data[start:start + BATCH_SIZE]
    if not batch:
        break

    instruction_text = """请输出 JSON（不要 markdown 代码块），每条为：
        {"Id": <int>, "Relation": <string>, "entity": [<string>, <string>, ...], "origin_entity": [<string>, <string>, ...]}

        任务：为每个输入问题抽取最核心的实体间关系短语 Relation，并提取相关的实体。关系短语应该能帮助检索系统更好地理解问题并获取相关证据。

        硬性规则（必须遵守）：

        **entity提取规则：**
        1) 在提取实体时，应确保完整提取涉及的所有相关实体。特别是人名、地名、组织名等多部分实体，不应漏掉任何部分。
        2) 对于人名、地名等复合实体，保留其全称，而不是单一部分。例如，“Nicole Plotzkuer”应作为一个完整实体，而不是简化为“Nicole”。
        3) 对于时间、地点、角色等背景信息，若它们对理解问题中的关系至关重要，应作为实体添加。例如，“Sistine Chapel Choir director/manager”作为一个实体，应包含完整的职位信息，不要拆分。
        4) 对于职位、角色、职称等，确保其完整性。类似于“Missouri Valley Conference Coach of the Year”应作为一个完整的实体，而不是拆分成“Coach”或“Missouri Valley Conference”。
        5) 对于复合实体（如“Steve Bannon”），确保其作为一个整体提取，避免拆分成“Steve”和“Bannon”。
        6) **`entity`** 不进行简化，必须保持详细且丰富的内容，包含所有背景信息。例如，类似“Coach of the Year”应保留其详细的描述，而不是简化为“Coach”。

        **relation提取规则：**
        7) Relation 只保留“关系谓词/关系短语”，用于后续检索/匹配；不要把整句 Question 复述进 Relation。
        8) 去掉所有冗余信息，但 **如果时间、地点、背景等信息对关系理解至关重要**，则可以保留。对于无法去除的背景信息，应避免过度简化，保持关系表达的完整性。
        9) 只保留能够连接“主实体 ↔ 目标信息/对象”的最短关系表达；优先保留动词或介词短语（如 founded / was born in / played for / headquartered in / filmed at）。但确保这不影响原始含义的完整性。
        10) 如果问题在问“谁/什么/哪里/哪个”之类答案本身，不要把疑问词放进 Relation（例如不要输出 "who" / "what" / "which"）。
        11) 如果问题是 “what was his/her position / role / title” 这类，Relation 用 "was his/her position" 或 "held the position of" 这类短关系即可。
        12) 保证简化后的 Relation 能在实际应用中有效地描述实体间的关系，避免信息过度丢失。
        13) 对于复杂的隐性时间问题，尽量保留时间相关的推理关系（例如“在...之后”或“在...期间”）。这类问题通常需要通过递归解析或中间问题生成来转换成显性时间，因此应确保时间约束清晰地反映在关系中，避免因简化失去必要的推理信息。
        14) **特别注意实体完整性：** 对于多部分实体（如人名或地名），应保留完整的实体信息，避免过度简化。确保提取到的关系包括完整的实体信息，不会因简化而丢失重要部分。

        输出示例：
        examples: [
          # 正面例子（改进后）：
          {
            "input": {
              "Id": 15953,
              "Question": "When Royce Waltman was named the Missouri Valley Conference Coach of the Year start date?"
            },
            "output": {
              "Id": 15953,    
              "Relation": "was named the Missouri Valley Conference Coach of the Year start date",
              "entity": ["Royce Waltman", "Missouri Valley Conference Coach of the Year", "Coach", "start", "named", "Missouri Valley Conference"],
              "origin_entity": ["Coach of the Year", "start", "named", "Missouri Valley Conference", "Royce Waltman", "Coach"]
            }
          },
          {
            "input": {
              "Id": 15954,
              "Question": "When Royce Waltman was named the Missouri Valley Conference Coach of the Year end date?"
            },
            "output": {
              "Id": 15954,
              "Relation": "was named the Missouri Valley Conference Coach of the Year end date",
              "entity": ["Royce Waltman", "Missouri Valley Conference Coach of the Year", "Coach", "end", "named", "Missouri Valley Conference"],
              "origin_entity": ["Coach of the Year", "end", "Royce Waltman", "Missouri Valley Conference", "Coach", "date", "year", "named"]
            }
          },
          {
            "input": {
              "Id": 15955,
              "Question": "When Steve Bannon finished his military service start date?"
            },
            "output": {
              "Id": 15955,
              "Relation": "finished his military service start date",
              "entity": ["Steve Bannon", "military service", "start"],
              "origin_entity": ["Bannon", "military service", "start", "Steve"]
            }
          },
          {
            "input": {
              "Id": 15956,
              "Question": "When Steve Bannon finished his military service end date?"
            },
            "output": {
              "Id": 15956,
              "Relation": "finished his military service end date",
              "entity": ["Steve Bannon", "military service", "end"],
              "origin_entity": ["Bannon", "military service", "end", "Steve"]
            }
          }
        ]"""

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
        if _id is not None and isinstance(rel, str):
            id2rel[int(_id)] = rel.strip()

    # 写回 batch，并更新 existing_by_id
    for item in batch:
        _id = int(item["Id"])
        if _id in id2rel:
            item["Relation"] = id2rel[_id]  # 写入简化后的
        # 合并到总结果（覆盖更新）
        existing_by_id[_id] = item

    # 每 10 条保存一次（同一个文件）
    merged = [existing_by_id[k] for k in sorted(existing_by_id.keys())]
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"[saved] {start}~{start+len(batch)-1}  (total saved: {len(merged)})")

print(f"Done. Output -> {OUTPUT_FILE}")
