import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from faith.library.utils import get_config  # noqa: E402
from faith.temporal_qu.seq2seq_tsf.seq2seq_tsf_module import Seq2SeqTSFModule  # noqa: E402


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    text = str(text).strip().lower()
    return " ".join(text.split())


def load_json(path: Path) -> Any:
    path = resolve_path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_id_map(
    data: List[Dict[str, Any]],
    id_key: str = "Id",
) -> Tuple[Dict[int, Dict[str, Any]], List[int]]:
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


def extract_relation_from_tsf(tsf_text: str, tsf_delimiter: str) -> str:
    if not tsf_text:
        return ""
    text = str(tsf_text).strip()

    candidate_delimiters = []
    if tsf_delimiter:
        candidate_delimiters.append(tsf_delimiter)
    for fallback in ["||", "<extra_id_0>"]:
        if fallback not in candidate_delimiters:
            candidate_delimiters.append(fallback)

    for delimiter in candidate_delimiters:
        slots = text.split(delimiter)
        if len(slots) >= 2:
            relation = slots[1].strip()
            if relation:
                return relation

    return ""


def evaluate_predictions(
    gold_data: List[Dict[str, Any]],
    pred_data: List[Dict[str, Any]],
) -> Dict[str, Any]:
    gold_map, gold_duplicate_ids = build_id_map(gold_data, id_key="Id")
    pred_map, pred_duplicate_ids = build_id_map(pred_data, id_key="Id")

    gold_ids = set(gold_map.keys())
    pred_ids = set(pred_map.keys())
    matched_ids = sorted(gold_ids & pred_ids)

    strict_match_count = 0
    gold_contains_pred_count = 0
    pred_contains_gold_count = 0
    mismatches = []

    for item_id in matched_ids:
        gold_item = gold_map[item_id]
        pred_item = pred_map[item_id]

        gold_relation_raw = gold_item.get("human_relation", "")
        pred_relation_raw = pred_item.get("pred_relation", pred_item.get("relation", ""))

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

        if not strict_equal:
            mismatches.append(
                {
                    "Id": item_id,
                    "Question": gold_item.get("Question", ""),
                    "human_relation_normalized": gold_relation,
                    "pred_relation_normalized": pred_relation,
                    "pred_tsf": pred_item.get("pred_tsf", ""),
                    "strict_equal": strict_equal,
                    "gold_contains_pred": gold_contains_pred,
                    "pred_contains_gold": pred_contains_gold,
                }
            )

    matched_count = len(matched_ids)
    return {
        "summary": {
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
            "missing_in_pred_ids": sorted(gold_ids - pred_ids),
            "extra_in_pred_ids": sorted(pred_ids - gold_ids),
        },
        "mismatches": mismatches,
    }


def run_inference(
    config_path: Path,
    input_path: Path,
    predictions_path: Path,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    config_path = resolve_path(config_path)
    input_path = resolve_path(input_path)
    predictions_path = resolve_path(predictions_path)
    config = get_config(str(config_path))
    module = Seq2SeqTSFModule(config)
    tsf_delimiter = config["tsf_delimiter"]

    gold_data = load_json(input_path)
    if not isinstance(gold_data, list):
        raise ValueError(f"Input file is not a JSON list: {input_path}")

    if limit is not None:
        gold_data = gold_data[:limit]

    predictions = []
    total = len(gold_data)
    for idx, item in enumerate(gold_data, 1):
        question = item.get("Question")
        item_id = item.get("Id")
        if question is None or item_id is None:
            continue

        pred_tsf = module.inference_on_question(question)
        pred_relation = extract_relation_from_tsf(pred_tsf, tsf_delimiter)

        predictions.append(
            {
                "Id": int(item_id),
                "Question": question,
                "human_relation": item.get("human_relation", ""),
                "original_relation": item.get("relation", ""),
                "pred_tsf": pred_tsf,
                "pred_relation": pred_relation,
            }
        )

        if idx % 5 == 0 or idx == total:
            print(f"[progress] {idx}/{total}")

    save_json(predictions_path, predictions)
    return predictions


def main():
    parser = argparse.ArgumentParser(
        description="Run a trained TSF model on human-checked data and evaluate relation accuracy."
    )
    parser.add_argument(
        "--config",
        type=Path,
        # default=Path("config/tiq/train_tqu_fer.yml"),
        default=Path("config/tiq/train_tqu_fer_pegasus_exp.yml"),
        help="Path to the TSF training/eval config.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("_data/tiq/tsf_annotation/322_50_1_weici_human_check.json"),
        # default=Path("_data/tiq/tsf_annotation/322_50_2_human_check.json"),
        help="Path to the human-checked JSON file.",
    )
    parser.add_argument(
        "--predictions-output",
        type=Path,
        default=Path("_data/tiq/tsf_annotation/humancheck_result/324_tsf_model_pegasus-large-predictions.json"),
        help="Where to save per-item TSF predictions.",
    )
    parser.add_argument(                                       ##########################################################  保存文件名
        "--report-output",
        type=Path,
        default=Path("_data/tiq/tsf_annotation/humancheck_result/324_tsf_model_pegasus-large1.json"),
        help="Where to save the aggregated evaluation report.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for a quick smoke test.",
    )
    args = parser.parse_args()

    args.config = resolve_path(args.config)
    args.input = resolve_path(args.input)
    args.predictions_output = resolve_path(args.predictions_output)
    args.report_output = resolve_path(args.report_output)

    gold_data = load_json(args.input)
    if not isinstance(gold_data, list):
        raise ValueError(f"Input file is not a JSON list: {args.input}")

    predictions = run_inference(
        config_path=args.config,
        input_path=args.input,
        predictions_path=args.predictions_output,
        limit=args.limit,
    )

    eval_gold_data = gold_data[: args.limit] if args.limit is not None else gold_data
    report = evaluate_predictions(eval_gold_data, predictions)
    report["summary"]["config_path"] = str(args.config.resolve())
    report["summary"]["input_path"] = str(args.input.resolve())
    report["summary"]["predictions_output"] = str(args.predictions_output.resolve())
    report["summary"]["report_output"] = str(args.report_output.resolve())

    save_json(args.report_output, report)

    print("Evaluation finished.")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
