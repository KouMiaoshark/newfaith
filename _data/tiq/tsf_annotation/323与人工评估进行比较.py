import os
import json
import argparse
from typing import Any, Dict, List, Tuple


def normalize_text(text: Any) -> str:
    """
    标准化字符串：
    - None -> ""
    - 转小写
    - 去首尾空格
    - 连续空白压成一个空格
    """
    if text is None:
        return ""
    text = str(text).strip().lower()
    text = " ".join(text.split())
    return text


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def build_id_map(
    data: List[Dict[str, Any]],
    id_key: str = "Id",
) -> Tuple[Dict[int, Dict[str, Any]], List[int]]:
    """
    构建 Id -> item 映射。
    若出现重复 Id，只保留第一次出现的记录，同时返回重复 Id 列表。
    """
    id_map = {}
    duplicate_ids = []

    for item in data:
        if id_key not in item:
            continue
        try:
            item_id = int(item[id_key])
        except Exception:
            continue

        if item_id in id_map:
            duplicate_ids.append(item_id)
        else:
            id_map[item_id] = item

    return id_map, duplicate_ids


def main():
    parser = argparse.ArgumentParser(
        description="使用 human_relation 评估 syntax_relation_train.json 中 relation 的正确率。"
    )
    parser.add_argument(
        "--gold",
        type=str,
        # default="322_50_1_weici_human_check.json",
        default="322_50_2_human_check.json",
        help="人工标注文件路径（包含 human_relation）",
    )
    parser.add_argument(
        "--pred",
        type=str,
        # default="syntax_relation_train.json",
        # default="322syntax_relation_train.json",
        default="324移除句法分析测试性能变化_无the.json",
        # default="227dep_relation_train_6000.json",

        help="预测文件路径（包含 relation）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join("humancheck_result", "移除句法分析_无the_check2.json"),
        help="输出结果文件路径",
    )

    args = parser.parse_args()

    gold_path = args.gold
    pred_path = args.pred
    output_path = args.output

    if not os.path.exists(gold_path):
        raise FileNotFoundError(f"找不到 gold 文件: {gold_path}")
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"找不到 pred 文件: {pred_path}")

    gold_data = load_json(gold_path)
    pred_data = load_json(pred_path)

    if not isinstance(gold_data, list):
        raise ValueError(f"gold 文件不是 JSON 数组: {gold_path}")
    if not isinstance(pred_data, list):
        raise ValueError(f"pred 文件不是 JSON 数组: {pred_path}")

    gold_map, gold_duplicate_ids = build_id_map(gold_data, id_key="Id")
    pred_map, pred_duplicate_ids = build_id_map(pred_data, id_key="Id")

    gold_ids = set(gold_map.keys())
    pred_ids = set(pred_map.keys())

    matched_ids = sorted(gold_ids & pred_ids)
    missing_in_pred = sorted(gold_ids - pred_ids)
    extra_in_pred = sorted(pred_ids - gold_ids)

    strict_match_count = 0
    gold_contains_pred_count = 0
    pred_contains_gold_count = 0

    comparisons = []
    mismatches = []

    for item_id in matched_ids:
        gold_item = gold_map[item_id]
        pred_item = pred_map[item_id]

        question = gold_item.get("Question", "")

        gold_relation_raw = gold_item.get("human_relation", "")
        pred_relation_raw = pred_item.get("relation", "")

        gold_relation = normalize_text(gold_relation_raw)
        pred_relation = normalize_text(pred_relation_raw)

        strict_equal = gold_relation == pred_relation
        gold_contains_pred = (pred_relation != "") and (pred_relation in gold_relation)
        pred_contains_gold = (gold_relation != "") and (gold_relation in pred_relation)

        if strict_equal:
            strict_match_count += 1
        if gold_contains_pred:
            gold_contains_pred_count += 1
        if pred_contains_gold:
            pred_contains_gold_count += 1

        record = {
            "Id": item_id,
            "Question": question,
            # "human_relation_raw": gold_relation_raw,
            # "pred_relation_raw": pred_relation_raw,
            "human_relation_normalized": gold_relation,
            "pred_relation_normalized": pred_relation,
            "strict_equal": strict_equal,
            "gold_contains_pred": gold_contains_pred,
            "pred_contains_gold": pred_contains_gold,
        }
        comparisons.append(record)

        if not strict_equal:
            mismatches.append(record)

    matched_count = len(matched_ids)

    result = {
        "summary": {
            "gold_file": os.path.abspath(gold_path),
            "pred_file": os.path.abspath(pred_path),
            "output_file": os.path.abspath(output_path),
            "gold_total_items": len(gold_data),
            "pred_total_items": len(pred_data),
            "gold_unique_ids": len(gold_ids),
            "pred_unique_ids": len(pred_ids),
            "matched_by_id": matched_count,
            "gold_duplicate_id_count": len(gold_duplicate_ids),
            "pred_duplicate_id_count": len(pred_duplicate_ids),
            "strict_match_count": strict_match_count,
            "gold_contains_pred_count": gold_contains_pred_count,
            "pred_contains_gold_count": pred_contains_gold_count,
            "strict_accuracy": round(strict_match_count / matched_count, 6) if matched_count else 0.0,
            "gold_contains_pred_rate": round(gold_contains_pred_count / matched_count, 6) if matched_count else 0.0,
            "pred_contains_gold_rate": round(pred_contains_gold_count / matched_count, 6) if matched_count else 0.0,
        },
        # "id_check": {
        #     "missing_in_pred_ids": missing_in_pred,
        # },
        "mismatches": mismatches,
    }

    output_dir = os.path.dirname(output_path)
    if output_dir:
        ensure_dir(output_dir)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("评估完成。")
    print(f"结果已保存到: {os.path.abspath(output_path)}")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()