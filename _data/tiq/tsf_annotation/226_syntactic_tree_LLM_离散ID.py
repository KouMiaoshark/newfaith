import os
import json
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI

# =========================================================
# 运行配置（在 PyCharm 里改这里即可）
# =========================================================
API_KEY = "sk-963eb925b9954830a0b817307e0a449b"     # <<< 替换成你的 key（不要提交到 git）
BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-chat"

# INPUT_PATH = "simplified_dev.json"
INPUT_PATH = "simplified_train.json"
OUTPUT_PATH = "324移除句法分析测试性能变化_无the.json"

BATCH_SIZE = 30

# 处理范围控制（三选一/组合）
MAX_N = None            # 只处理前 MAX_N 条；None=全量
QUESTION_NUM = None      # 只处理最后 QUESTION_NUM 条；None=不用这个逻辑
START_INDEX = 5590      # 手动指定起始 index（优先级最高）
END_INDEX = 6000        # 手动指定结束 index（不含）

# 单条调试模式（需要时打开）
RUN_SINGLE = False
SINGLE_QUESTION = "Which international football team did Roy Keane join after being a member of the Republic of Ireland national under-21 football team?"
SINGLE_ID = 1
# =========================================================


def strip_code_fences(text: str) -> str:
    """去掉 ```json ... ``` 或 ``` ... ``` 包裹"""
    t = (text or "").strip()
    if t.startswith("```"):
        first_nl = t.find("\n")
        if first_nl != -1:
            t = t[first_nl + 1 :]
        t = t.strip()
        if t.endswith("```"):
            t = t[:-3].strip()
    return t.strip()


def parse_any_json_objects(text: str) -> List[Any]:
    """
    尽可能把模型输出解析成 JSON 对象列表：
    - 标准 JSON 数组
    - 单个 JSON 对象
    - 连续多个 JSON 对象（raw_decode 循环）
    """
    t = strip_code_fences(text)

    # 1) 直接 loads
    try:
        obj = json.loads(t)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            return [obj]
    except Exception:
        pass

    # 2) raw_decode 连续解析
    decoder = json.JSONDecoder()
    i = 0
    objs = []
    while i < len(t):
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


def build_instruction_text() -> str:
    """
    让 LLM 输出：
    [{"Id":..., "relation": "..."} , ...]
    relation =主句中的核心关系表达
    """
    examples = [
        {
            "input": {
                "Id": 6005,
                "Question": "Which international football team did Roy Keane join after being a member of the Republic of Ireland national under-21 football team?"
            },
            "output": {
                "Id": 6005,
                "relation": "join"
            }
        },

        {
            "input": {
                "Id": 6017,
                "Question": "During Ashley Cooper's tennis career, in which Grand Slam events did he win his three singles titles?"
            },
            "output": {
                "Id": 6017,
                "relation": "win"
            }
        },

        {
            "input": {
                "Id": 6071,
                "Question": "What ethnic group did Milo O'Shea belong to before he passed away in Manhattan, New York at the age of 86?"
            },
            "output": {
                "Id": 6071,
                "relation": "belong to"
            }
        },

        {
            "input": {
                "Id": 6084,
                "Question": "When Alice Belin Du Pont was married to Pierre Samuel Du Pont, which car company did he become a director of?"
            },
            "output": {
                "Id": 6084,
                "relation": "become a director of"
            }
        },

        {
            "input": {
                "Id": 6110,
                "Question": "During Jason Kenney's leadership of the Progressive Conservative Association of Alberta, for which elected legislative house was he the member from Calgary-Lougheed?"
            },
            "output": {
                "Id": 6110,
                "relation": "the member from"
            }
        },

        {
            "input": {
                "Id": 6082,
                "Question": "What organization was Bolivia a member of when it became the first South American country to declare the right of indigenous people to govern themselves?"
            },
            "output": {
                "Id": 6082,
                "relation": "a member of"
            }
        },

        {
            "input": {
                "Id": 6089,
                "Question": "At the time of Central African Republic's independence ceremony for Bokassa I, which international organizations was it a member of?"
            },
            "output": {
                "Id": 6089,
                "relation": "a member of"
            }
        },

        {
            "input": {
                "Id": 6092,
                "Question": "During the time of Portugal's current constitution, which intergovernmental organisation was it a member of?"
            },
            "output": {
                "Id": 6092,
                "relation": "a member of"
            }
        }
    ]

    instruction = f"""  
    你是一个英文 relation 抽取器。请对给定的英文问题，从主句中抽取核心关系表达（core relation expression）作为 relation。  

    relation 的定义：  
    - relation 不是整句复述，而是主句主干中的核心关系表达。  
    - relation 优先来自主句中的连续原文片段。  
    - relation 可以是：  
      - 实义动词（如 "join", "receive", "purchase"）  
      - 动词短语（如 "was founded", "was leading", "complying with"）  
      - 表语名词/名词短语（如 "the wife of", "a member of", "the head coach of"）  
      - 表语形容词/补足结构（如 "was true regarding"）  
    - 不要求 relation 必须是单个谓词动词；关键是它要能表达问题主句的核心关系。  

    输出要求（必须严格遵守）：  
    1) 只输出 JSON（不要 markdown 代码块，不要解释，不要多余文本）。  
    2) 输出为 JSON 数组，每个元素格式严格为：  
    {{"Id": <int>, "relation": <string>}}  
    3) 请确保输出中的 Id 与输入完全一致。  

    relation 抽取规则（非常重要）：  
    1. 先识别并排除以下成分，不要把它们放入 relation：  
    - 疑问词和答案壳：who, what, which ..., when, where 等，以及它们引导的疑问名词短语  
    - 前置或后置时间/背景说明：when ..., before ..., after ..., during ..., according to ..., 等  
    - 与主句核心关系无关的附加说明、比赛场景、背景状语等  

    2. relation 必须来自“去掉上述壳层后”的主句主干。  

    3. 优先选择主句中的连续原文片段，不要自行改写、概括、补词或跨位置拼接。  
    - 能连续截取就连续截取  
    - 如果不能自然连续截取，则宁可短一点，也不要强行组合  
    - 如果原句中的连续 relation 片段包含冠词（the, a, an），且该冠词属于该关系短语的自然组成部分，则必须保留，不能省略  
    - 尤其不要把 "the wife of" 改成 "wife of"，也不要把 "the head coach of" 改成 "head coach of"  
    - 总原则：relation 应尽量保持为原句中的连续原文片段，而不是删去冠词后的“压缩版”  

    4. 对普通动词句：  
    - 优先抽取核心实义动词  
    - 若动词后带有不可缺少的介词/小品词/补足成分，可一并保留  
    - 例如："join", "purchase", "awarded", "complying with", "become the coach of"  

    5. 对系动词结构（be + 表语）：  
    - 不要只输出 "be"/"was"/"is"  
    - 应抽取表语中心及其必要补足成分  
    - 例如：  
      - "was a member of"  
      - "was the head coach of"  
      - "was true regarding"  

    6. 若 relation 中的介词是关系短语的一部分，则应保留：  
    - 如 "a member of", "the head coach of", "complying with", "true regarding"  
    - 但若只是外围时间/地点/背景介词短语，则不要保留  

    7. relation 一律小写，去掉问号等标点，保留原句核心词序，不要包含 wh-词。  
    8. relation 应尽量简短，但以完整表达核心关系为准；不强制限制为 1~4 个 token。  
    9. 如果主句是被动结构，可保留被动核心短语，如 "was founded"。  
    10. 如果主句中职位、身份、头衔等是关系不可缺的一部分，应整体保留到语义完整为止，包括其中原句自带的冠词。  

    示例（仅展示格式）：  
    {json.dumps(examples, ensure_ascii=False, indent=2)}  

    现在请处理我随后给出的输入（将是一个 JSON 数组，元素包含 Id 和 Question）。  
    """.strip()
    return instruction
#    - 不要为了让 relation 更短而删除原句中本来连续出现的冠词；如果冠词属于关系短语本身，就必须保留。

# def normalize_one(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
#     if not isinstance(obj, dict):
#         return None
#     _id = obj.get("Id", obj.get("id"))
#     if _id is None:
#         return None
#
#     rel = obj.get("relation") or obj.get("Relation") or obj.get("predicate") or obj.get("Predicate")
#     tree = obj.get("syntax_tree") or obj.get("parse_tree") or obj.get("tree") or obj.get("syntaxTree")
#
#     if rel is None or tree is None:
#         return None
#
#     try:
#         _id = int(_id)
#     except Exception:
#         return None
#
#     rel = str(rel).strip()
#     tree = str(tree).strip()
#
#     if not rel or not tree:
#         return None
#
#     return {"Id": _id, "relation": rel, "syntax_tree": tree}

def normalize_one(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(obj, dict):
        return None

    _id = obj.get("Id", obj.get("id"))
    if _id is None:
        return None

    rel = obj.get("relation") or obj.get("Relation") or obj.get("predicate") or obj.get("Predicate")
    if rel is None:
        return None

    try:
        _id = int(_id)
    except Exception:
        return None

    rel = str(rel).strip()
    if not rel:
        return None

    return {"Id": _id, "relation": rel}


def call_llm_for_batch(client: OpenAI, model: str, batch: List[Dict[str, Any]], max_retries: int = 3) -> List[Dict[str, Any]]:
    instruction_text = build_instruction_text()
    payload = [{"Id": int(x["Id"]), "Question": x["Question"]} for x in batch]

    messages = [
        {"role": "system", "content": "You are a precise syntactic parser and predicate extractor."},
        {"role": "user", "content": instruction_text},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=False,
                temperature=0
            )
            raw = resp.choices[0].message.content
            objs = parse_any_json_objects(raw)
            out: List[Dict[str, Any]] = []
            for o in objs:
                norm = normalize_one(o)
                if norm:
                    out.append(norm)
            return out
        except Exception as e:
            last_err = e
            time.sleep(1.5 * attempt)

    raise RuntimeError(f"LLM 调用失败（已重试 {max_retries} 次）：{last_err}")


def load_json_list(path: str) -> List[Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return obj if isinstance(obj, list) else []


def compute_range(total: int) -> (int, int):
    """
    计算 [start_index, end_index)：
    START_INDEX/END_INDEX 优先；
    否则用 QUESTION_NUM + MAX_N 的组合逻辑；
    """
    # end_index
    if MAX_N is None:
        max_n = total
    else:
        max_n = min(MAX_N, total)

    if END_INDEX is not None:
        end_index = max(0, min(END_INDEX, total))
    else:
        end_index = max_n

    # start_index
    if START_INDEX is not None:
        start_index = max(0, min(START_INDEX, total))
    elif QUESTION_NUM is not None:
        start_index = max(0, end_index - QUESTION_NUM)
    else:
        start_index = 0

    if start_index >= end_index:
        raise RuntimeError(f"无效区间：start_index={start_index}, end_index={end_index}, total={total}")
    return start_index, end_index


def run_single(client: OpenAI):
    one = [{"Id": int(SINGLE_ID), "Question": SINGLE_QUESTION}]
    out = call_llm_for_batch(client, MODEL, one)
    if not out:
        raise RuntimeError("未得到有效输出，请检查提示词或模型返回格式。")
    print(json.dumps(out[0], ensure_ascii=False, indent=2))






TARGET_IDS = {5240,913,205,2007,1829,1144,840,5545,4469,713,4839,3458,261,245,768,1792,1906,4141,4933,218,4599,1629,5867,5325,5747,4466,3438,1806,3681,4829,2279,54,1308,5721,3464,2789,2277,1274,1764,2759,838,760,3114,793,2942,2819,4947,2167,356,316
}
def run_file(client: OpenAI):
    data = load_json_list(INPUT_PATH)
    if not data:
        raise RuntimeError(f"输入文件为空或不是 JSON list：{INPUT_PATH}")

    total = len(data)

    # 断点续跑：读取已有输出
    existing_by_id: Dict[int, Dict[str, Any]] = {}
    if os.path.exists(OUTPUT_PATH):
        try:
            existing = load_json_list(OUTPUT_PATH)
            for item in existing:
                norm = normalize_one(item) if isinstance(item, dict) else None
                if norm:
                    existing_by_id[norm["Id"]] = norm
        except Exception:
            pass

    t0 = time.time()

    # =========================
    # 新增：优先按 TARGET_IDS 精确补跑
    # =========================
    if TARGET_IDS is not None:
        target_ids = {int(x) for x in TARGET_IDS}

        selected = []
        found_ids = set()

        for x in data:
            if not isinstance(x, dict):
                continue
            if "Id" not in x or "Question" not in x:
                continue

            _id = int(x["Id"])
            if _id in target_ids:
                found_ids.add(_id)
                # 已经在输出文件里的就跳过，避免重复请求
                if _id not in existing_by_id:
                    selected.append({"Id": _id, "Question": x["Question"]})

        missing_in_input = sorted(target_ids - found_ids)

        print(f"[config] total={total}, target_ids={sorted(target_ids)}, saved={len(existing_by_id)}")
        if missing_in_input:
            print(f"[warning] 以下 TARGET_IDS 在输入文件中未找到: {missing_in_input}")

        if not selected:
            print("没有需要补跑的新题目（可能都已存在于输出文件中，或输入文件里没有这些 Id）。")
            return

        # 按 BATCH_SIZE 分批
        for start in range(0, len(selected), BATCH_SIZE):
            batch = selected[start: start + BATCH_SIZE]
            ids = [b["Id"] for b in batch]
            print(f"[batch] target batch {start}~{start+len(batch)-1} | new={len(batch)} | ids={ids}")

            out_items = call_llm_for_batch(client, MODEL, batch)

            got = 0
            for o in out_items:
                existing_by_id[o["Id"]] = o
                got += 1

            merged = [existing_by_id[k] for k in sorted(existing_by_id.keys())]
            with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
                json.dump(merged, f, ensure_ascii=False, indent=2)

            elapsed = time.time() - t0
            print(f"[saved] got={got}/{len(batch)} | total_saved={len(merged)} | elapsed={elapsed:.1f}s -> {OUTPUT_PATH}")

        print(f"Done. Output -> {OUTPUT_PATH}")
        return

    # =========================
    # 原来的区间模式（只有 TARGET_IDS 为 None 时才走这里）
    # =========================
    start_index, end_index = compute_range(total)
    print(f"[config] total={total}, range=[{start_index}, {end_index}), batch_size={BATCH_SIZE}, saved={len(existing_by_id)}")

    for start in range(start_index, end_index, BATCH_SIZE):
        batch_raw = data[start: min(start + BATCH_SIZE, end_index)]

        batch = []
        for x in batch_raw:
            if not isinstance(x, dict):
                continue
            if "Id" not in x or "Question" not in x:
                continue
            _id = int(x["Id"])
            if _id in existing_by_id:
                continue
            batch.append({"Id": _id, "Question": x["Question"]})

        if not batch:
            continue

        ids = [b["Id"] for b in batch]
        print(f"[batch] idx {start}~{start+len(batch_raw)-1} | new={len(batch)} | ids={ids[:5]}{'...' if len(ids)>5 else ''}")

        out_items = call_llm_for_batch(client, MODEL, batch)

        got = 0
        for o in out_items:
            existing_by_id[o["Id"]] = o
            got += 1

        merged = [existing_by_id[k] for k in sorted(existing_by_id.keys())]
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)

        elapsed = time.time() - t0
        print(f"[saved] got={got}/{len(batch)} | total_saved={len(merged)} | elapsed={elapsed:.1f}s -> {OUTPUT_PATH}")

    print(f"Done. Output -> {OUTPUT_PATH}")

def main():
    if not API_KEY or API_KEY.startswith("sk-REPLACE"):
        raise RuntimeError("请先在代码开头把 API_KEY 替换成真实 key。")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    if RUN_SINGLE:
        run_single(client)
    else:
        run_file(client)


if __name__ == "__main__":
    main()