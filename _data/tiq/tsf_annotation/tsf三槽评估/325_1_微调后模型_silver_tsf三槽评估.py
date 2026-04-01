import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


SLOT_NAMES = ("answer_type", "temporal_signal", "categorization")


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_json(path: Path) -> Any:
    path = resolve_path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        parts = [normalize_text(item) for item in value]
        parts = [part for part in parts if part]
        return " | ".join(parts)
    text = str(value).strip().lower()
    return " ".join(text.split())


def normalize_categorization_for_eval(value: Any) -> str:
    normalized = normalize_text(value)
    return "implicit" if normalized == "implicit" else "none"


def is_slot_correct(slot_name: str, pred_value: Any, gold_value: Any) -> bool:
    if slot_name == "categorization":
        return normalize_categorization_for_eval(pred_value) == normalize_categorization_for_eval(
            gold_value
        )
    return normalize_text(pred_value) == normalize_text(gold_value)


def parse_tsf_slots(tsf_text: Any, tsf_delimiter: str) -> Dict[str, str]:
    text = "" if tsf_text is None else str(tsf_text).strip()
    candidate_delimiters = []
    if tsf_delimiter:
        candidate_delimiters.append(tsf_delimiter)
    for fallback in ("||", "<extra_id_0>"):
        if fallback not in candidate_delimiters:
            candidate_delimiters.append(fallback)

    slots: Optional[List[str]] = None
    for delimiter in candidate_delimiters:
        split_slots = text.split(delimiter, 4)
        if len(split_slots) > 1:
            slots = split_slots
            break

    if slots is None:
        slots = [text]

    slots = [slot.strip() for slot in slots]
    if len(slots) < 5:
        slots.extend([""] * (5 - len(slots)))
    else:
        slots = slots[:5]

    return {
        "entity": slots[0],
        "relation": slots[1],
        "answer_type": slots[2],
        "temporal_signal": slots[3],
        "categorization": slots[4],
    }


def extract_gold_slots(item: Dict[str, Any]) -> Dict[str, str]:
    silver_tsf = item.get("silver_tsf", {})
    if not isinstance(silver_tsf, dict):
        silver_tsf = {}
    return {slot_name: normalize_text(silver_tsf.get(slot_name, "")) for slot_name in SLOT_NAMES}


def has_usable_silver_tsf(item: Dict[str, Any]) -> bool:
    silver_tsf = item.get("silver_tsf")
    if not isinstance(silver_tsf, dict) or not silver_tsf:
        return False
    return any(normalize_text(silver_tsf.get(slot_name, "")) for slot_name in SLOT_NAMES)


def in_selected_id_range(item_id: int, id_range: str) -> bool:
    if id_range == "all":
        return True
    if id_range == "0-6000":
        return 0 <= item_id <= 6000
    if id_range == "6000-8000":
        return 6000 <= item_id <= 8000
    raise ValueError(f"Unsupported id_range: {id_range}")


def run_inference(
    config_path: Path,
    input_path: Path,
    predictions_path: Path,
    id_range: str = "all",
    limit: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    from faith.library.utils import get_config
    from faith.temporal_qu.seq2seq_tsf.seq2seq_tsf_module import Seq2SeqTSFModule

    config_path = resolve_path(config_path)
    input_path = resolve_path(input_path)
    predictions_path = resolve_path(predictions_path)

    config = get_config(str(config_path))
    module = Seq2SeqTSFModule(config)
    tsf_delimiter = config["tsf_delimiter"]

    data = load_json(input_path)
    if not isinstance(data, list):
        raise ValueError(f"Input file is not a JSON list: {input_path}")

    filtered_data = []
    skipped_empty_silver_tsf = 0
    for item in data:
        item_id = item.get("Id")
        try:
            item_id_int = int(item_id)
        except Exception:
            continue
        if not has_usable_silver_tsf(item):
            skipped_empty_silver_tsf += 1
            continue
        if in_selected_id_range(item_id_int, id_range):
            filtered_data.append(item)

    data = filtered_data
    if limit is not None:
        data = data[:limit]

    predictions: List[Dict[str, Any]] = []
    total = len(data)

    for idx, item in enumerate(data, 1):
        question = item.get("Question") or item.get("question")
        item_id = item.get("Id")

        if question is None or item_id is None:
            continue

        pred_tsf = module.inference_on_question(question)
        pred_slots_raw = parse_tsf_slots(pred_tsf, tsf_delimiter)
        pred_slots = {slot_name: normalize_text(pred_slots_raw.get(slot_name, "")) for slot_name in SLOT_NAMES}
        gold_slots = extract_gold_slots(item)
        slot_correct = {
            slot_name: is_slot_correct(slot_name, pred_slots[slot_name], gold_slots[slot_name])
            for slot_name in SLOT_NAMES
        }

        predictions.append(
            {
                "Id": int(item_id),
                "Question": question,
                "pred_tsf": pred_tsf,
                "pred_slots_raw": pred_slots_raw,
                "pred_slots": pred_slots,
                "gold_slots": gold_slots,
                "slot_correct": slot_correct,
                "all_three_correct": all(slot_correct.values()),
            }
        )

        if idx % 5 == 0 or idx == total:
            print(f"[progress] {idx}/{total}")

    save_json(predictions_path, predictions)
    meta = {
        "skipped_empty_silver_tsf_count": skipped_empty_silver_tsf,
    }
    return predictions, meta


def evaluate_predictions(predictions: List[Dict[str, Any]], meta: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
    slot_stats = {
        slot_name: {
            "correct": 0,
            "total": 0,
            "accuracy": 0.0,
        }
        for slot_name in SLOT_NAMES
    }

    all_three_correct_count = 0
    mismatches = []

    for item in predictions:
        gold_slots = item["gold_slots"]
        pred_slots = item["pred_slots"]
        slot_correct = {
            slot_name: is_slot_correct(slot_name, pred_slots.get(slot_name, ""), gold_slots.get(slot_name, ""))
            for slot_name in SLOT_NAMES
        }
        item["slot_correct"] = slot_correct
        item["all_three_correct"] = all(slot_correct.values())
        item_has_mismatch = False
        for slot_name in SLOT_NAMES:
            slot_stats[slot_name]["total"] += 1
            if slot_correct[slot_name]:
                slot_stats[slot_name]["correct"] += 1
            else:
                item_has_mismatch = True

        if item["all_three_correct"]:
            all_three_correct_count += 1

        if item_has_mismatch:
            mismatches.append(
                {
                    "Id": item["Id"],
                    "Question": item["Question"],
                    "pred_tsf": item["pred_tsf"],
                    "gold_slots": item["gold_slots"],
                    "pred_slots": item["pred_slots"],
                    "slot_correct": item["slot_correct"],
                }
            )

    total_items = len(predictions)
    meta = meta or {}
    skipped_empty_silver_tsf_count = int(meta.get("skipped_empty_silver_tsf_count", 0))

    for slot_name in SLOT_NAMES:
        total = slot_stats[slot_name]["total"]
        correct = slot_stats[slot_name]["correct"]
        slot_stats[slot_name]["accuracy"] = round(correct / total, 6) if total else 0.0

    summary = {
        "total_items": total_items,
        "skipped_empty_silver_tsf_count": skipped_empty_silver_tsf_count,
        "all_three_correct_count": all_three_correct_count,
        "all_three_accuracy": round(all_three_correct_count / total_items, 6) if total_items else 0.0,
        "slot_metrics": slot_stats,
    }

    return {
        "summary": summary,
        "mismatches": mismatches,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned TSF model outputs against silver_tsf answer_type, temporal_signal, and categorization."
    )
    parser.add_argument(
        "--config",
        type=Path,
        # default=Path("config/tiq/train_tqu_fer_pegasus_exp.yml"),
        default=Path("config/tiq/train_tqu_fer.yml"),
        help="Path to the TSF training/eval config.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("_data/tiq/tsf_annotation/annotated_dev_faith.json"),
        help="Path to the annotated dev dataset.",
    )
    parser.add_argument(
        "--predictions-output",
        type=Path,
        default=Path("_data/tiq/tsf_annotation/tsf三槽评估/annotated_dev_faith_tsf_slot_bartbase_包含显性_predictions.json"),
        help="Where to save per-item predictions and slot comparisons.",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path("_data/tiq/tsf_annotation/tsf三槽评估/annotated_dev_faith_tsf_slot_bartbase_包含显性_report.json"),
        help="Where to save aggregated slot evaluation results.",
    )
    parser.add_argument(
        "--id-range",
        choices=["all", "0-6000", "6000-8000"],
        default="all",
        help="Only evaluate items whose Id falls in the selected inclusive range.",
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

    predictions, meta = run_inference(
        config_path=args.config,
        input_path=args.input,
        predictions_path=args.predictions_output,
        id_range=args.id_range,
        limit=args.limit,
    )

    report = evaluate_predictions(predictions, meta=meta)
    report["summary"]["id_range"] = args.id_range
    report["summary"]["config_path"] = str(args.config.resolve())
    report["summary"]["input_path"] = str(args.input.resolve())
    report["summary"]["predictions_output"] = str(args.predictions_output.resolve())
    report["summary"]["report_output"] = str(args.report_output.resolve())

    save_json(args.report_output, report)

    print("Evaluation finished.")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
