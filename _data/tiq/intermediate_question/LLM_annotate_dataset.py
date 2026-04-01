import os
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from openai import OpenAI

# =========================================================
# 运行配置（在 PyCharm 里改这里即可）
# =========================================================
API_KEY = "sk-963eb925b9954830a0b817307e0a449b"
BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-chat"

INPUT_PATH = "simplified_dev_ques.json"
OUTPUT_PATH = "implicit_intermediate_question_dev.json"

BATCH_SIZE = 30
MAX_WORKERS = 10

# 处理范围控制（三选一/组合）
MAX_N = None
QUESTION_NUM = None
START_INDEX = None
END_INDEX = None

# 单条调试模式（需要时打开）
RUN_SINGLE = False
SINGLE_QUESTION = "Who was the second director of the Isabella Stewart Gardner Museum when it was built"
SINGLE_ID = 1
# =========================================================


def strip_code_fences(text: str) -> str:
    """去掉 ```json ... ``` 或 ``` ... ``` 包裹"""
    t = (text or "").strip()
    if t.startswith("```"):
        first_nl = t.find("\n")
        if first_nl != -1:
            t = t[first_nl + 1:]
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

    try:
        obj = json.loads(t)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            return [obj]
    except Exception:
        pass

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
    [{"Id":..., "intermediate_question": "...", "answer_type": "..."} , ...]
    intermediate_question = 将隐式时间约束改写成显式中间问题
    answer_type = date / time interval
    """
    examples = [
        {
            "input": {
                "Id": 1,
                "Question": "Who was the second director of the Isabella Stewart Gardner Museum when it was built"
            },
            "output": {
                "Id": 1,
                "intermediate_question": "When Isabella Stewart Gardner Museum was built",
                "answer_type": "time interval"
            }
        },
        {
            "input": {
                "Id": 2,
                "Question": "When Wendy Doniger was president of the Association for Asian Studies, what publishing house was she based in New York"
            },
            "output": {
                "Id": 2,
                "intermediate_question": "When Wendy Doniger was president of the Association for Asian Studies",
                "answer_type": "time interval"
            }
        },
        {
            "input": {
                "Id": 3,
                "Question": "What administrative entity was Ezhou in before Huanghu District became part of it"
            },
            "output": {
                "Id": 3,
                "intermediate_question": "When Huangzhou District became part of Ezhou",
                "answer_type": "date"
            }
        },
        {
            "input": {
                "Id": 4,
                "Question": "After Bud Yorkin became the producer of NBC's The Tony Martin Show, who was his spouse?"
            },
            "output": {
                "Id": 4,
                "intermediate_question": "When Bud Yorkin became the producer of NBC's The Tony Martin Show",
                "answer_type": "date"
            }
        },
        {
            "input": {
                "Id": 5,
                "Question": "What book did Ira Levin write that was adapted into a film during the same time he wrote the play Deathtrap"
            },
            "output": {
                "Id": 5,
                "intermediate_question": "When Ira Levin wrote the play Deathtrap",
                "answer_type": "date"
            }
        },
        {
            "input": {
                "Id": 6,
                "Question": "What basketball team was Nathaniel Clifton playing for when his  career history with the Rens began"
            },
            "output": {
                "Id": 6,
                "intermediate_question": "When Nathaniel Clifton's career history with the Rens began",
                "answer_type": "time interval"
            }
        },
        {
            "input": {
                "Id": 7,
                "Question": "What team did Stevica Ristic play for before joining Shonan Bellmare"
            },
            "output": {
                "Id": 7,
                "intermediate_question": "When Stevica Ristic joining Shonan Bellmare",
                "answer_type": "time interval"
            }
        },
        {
            "input": {
                "Id": 8,
                "Question": "Which album was released by the Smashing Pumpkins after Mike Byrne joined the band"
            },
            "output": {
                "Id": 8,
                "intermediate_question": "When Mike Byrne joined Smashing Pumpkins",
                "answer_type": "time interval"
            }
        }
    ]

    instruction = f"""
你是一个英文隐式时间问题转换器。

任务：
给定一个英文 temporal question，请识别其中“隐式时间约束部分”，并把它改写成一个显式的中间问题（intermediate question），同时判断这个中间问题的答案类型（answer_type）。

输出目标：
- intermediate_question：只针对原问题里的隐式时间部分生成一个显式问题
- answer_type：只能是以下二选一
  - "date"
  - "time interval"

核心要求：
1. intermediate_question 必须以 "When" 开头。
2. intermediate_question 应该只表达“隐式时间约束本身”，不要重复主问题的信息需求。
3. 不要回答原问题，不要输出原问题答案。
4. 不要生成 start date / end date。
   - 这里只输出基础中间问题。
   - 如果 answer_type 是 "time interval"，后续系统会再自动扩展成 start date / end date。
5. 尽量保持简洁、自然，可适度规范专有名词拼写。
6. 如果隐式时间约束对应一个事件发生时刻，answer_type 用 "date"。
7. 如果隐式时间约束对应一个持续时期、任职期、活动期、区间，answer_type 用 "time interval"。

输出要求（必须严格遵守）：
1) 只输出 JSON（不要 markdown 代码块，不要解释，不要多余文本）。
2) 输出为 JSON 数组，每个元素格式严格为：
{{"Id": <int>, "intermediate_question": <string>, "answer_type": <string>}}
3) 请确保输出中的 Id 与输入完全一致。
4) answer_type 只能是 "date" 或 "time interval"。

示例：
{json.dumps(examples, ensure_ascii=False, indent=2)}

现在请处理我随后给出的输入（将是一个 JSON 数组，元素包含 Id 和 Question）。
""".strip()
    return instruction


def normalize_one(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(obj, dict):
        return None

    _id = obj.get("Id", obj.get("id"))
    if _id is None:
        return None

    question = (
        obj.get("intermediate_question")
        or obj.get("Intermediate_question")
        or obj.get("explicit_question")
        or obj.get("question")
    )
    answer_type = (
        obj.get("answer_type")
        or obj.get("Answer_type")
        or obj.get("type")
    )

    if question is None or answer_type is None:
        return None

    try:
        _id = int(_id)
    except Exception:
        return None

    question = str(question).strip()
    answer_type = str(answer_type).strip().lower()

    if not question:
        return None

    if answer_type not in {"date", "time interval"}:
        return None

    return {
        "Id": _id,
        "intermediate_question": question,
        "answer_type": answer_type
    }


def call_llm_for_batch(
    client: OpenAI,
    model: str,
    batch: List[Dict[str, Any]],
    max_retries: int = 3
) -> List[Dict[str, Any]]:
    instruction_text = build_instruction_text()
    payload = [{"Id": int(x["Id"]), "Question": x["Question"]} for x in batch]

    messages = [
        {
            "role": "system",
            "content": "You are a precise temporal question reformulation assistant."
        },
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


def save_output(path: str, existing_by_id: Dict[int, Dict[str, Any]]):
    merged = [existing_by_id[k] for k in sorted(existing_by_id.keys())]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    return merged


def build_batches(
    data: List[Any],
    start_index: int,
    end_index: int,
    batch_size: int,
    existing_by_id: Dict[int, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    batches: List[Dict[str, Any]] = []
    for start in range(start_index, end_index, batch_size):
        batch_raw = data[start:min(start + batch_size, end_index)]

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

        batches.append(
            {
                "start": start,
                "end": start + len(batch_raw) - 1,
                "batch": batch,
                "ids": [b["Id"] for b in batch],
            }
        )
    return batches


def decide_range(total_len: int):
    if START_INDEX is not None:
        s = max(0, START_INDEX)
        e = total_len if END_INDEX is None else min(total_len, END_INDEX)
        return s, e

    if MAX_N is not None:
        return 0, min(MAX_N, total_len)

    if QUESTION_NUM is not None:
        s = max(0, total_len - QUESTION_NUM)
        return s, total_len

    return 0, total_len


def main():
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    if RUN_SINGLE:
        result = call_llm_for_batch(
            client,
            MODEL,
            [{"Id": SINGLE_ID, "Question": SINGLE_QUESTION}]
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    data = load_json_list(INPUT_PATH)
    if not data:
        raise ValueError(f"输入文件不是有效 JSON 数组或为空：{INPUT_PATH}")

    if os.path.exists(OUTPUT_PATH):
        existing = load_json_list(OUTPUT_PATH)
    else:
        existing = []

    existing_by_id: Dict[int, Dict[str, Any]] = {}
    for x in existing:
        if not isinstance(x, dict) or "Id" not in x:
            continue
        try:
            existing_by_id[int(x["Id"])] = x
        except Exception:
            pass

    start_index, end_index = decide_range(len(data))
    batches = build_batches(data, start_index, end_index, BATCH_SIZE, existing_by_id)

    print(f"总数据量: {len(data)}")
    print(f"本次处理区间: [{start_index}, {end_index})")
    print(f"待处理 batch 数: {len(batches)}")
    print(f"已有结果数: {len(existing_by_id)}")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_meta = {
            executor.submit(call_llm_for_batch, client, MODEL, item["batch"]): item
            for item in batches
        }

        for future in as_completed(future_to_meta):
            meta = future_to_meta[future]
            try:
                result_list = future.result()
                for item in result_list:
                    existing_by_id[int(item["Id"])] = item

                save_output(OUTPUT_PATH, existing_by_id)
                print(
                    f"完成 batch [{meta['start']}, {meta['end']}], "
                    f"问题数={len(meta['batch'])}, 返回数={len(result_list)}"
                )
            except Exception as e:
                print(
                    f"batch 失败 [{meta['start']}, {meta['end']}], "
                    f"ids={meta['ids']}, err={e}"
                )

    merged = save_output(OUTPUT_PATH, existing_by_id)
    print(f"全部完成，输出条数: {len(merged)}")
    print(f"输出文件: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()