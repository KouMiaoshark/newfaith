import csv
import json
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent
IMPLICIT_PATH = CURRENT_DIR / "implicit_intermediate_question_train.json"
GPT_PATH = CURRENT_DIR / "gpt_annotate_generate_question_train.json"
OUTPUT_JSON = CURRENT_DIR / "intermediate_question_diff_report.json"
OUTPUT_CSV = CURRENT_DIR / "intermediate_question_diff_report.csv"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def normalize_text(value):
    if value is None:
        return ""
    return " ".join(str(value).split()).strip()


def normalize_answer_type(value):
    return normalize_text(value).lower()


def build_implicit_index(data):
    index = {}
    for item in data:
        item_id = item.get("Id")
        if item_id is None:
            continue
        index[item_id] = {
            "intermediate_question": normalize_text(item.get("intermediate_question")),
            "answer_type": normalize_answer_type(item.get("answer_type")),
        }
    return index


def build_gpt_index(data):
    index = {}
    for item in data:
        item_id = item.get("Id")
        if item_id is None:
            continue

        generated = item.get("silver_generated_question") or []
        generated_question = generated[0] if len(generated) > 0 else ""
        answer_type = generated[1] if len(generated) > 1 else ""

        index[item_id] = {
            "question": normalize_text(item.get("Question")),
            "intermediate_question": normalize_text(generated_question),
            "answer_type": normalize_answer_type(answer_type),
        }
    return index


def compare_datasets():
    implicit_data = load_json(IMPLICIT_PATH)
    gpt_data = load_json(GPT_PATH)

    implicit_index = build_implicit_index(implicit_data)
    gpt_index = build_gpt_index(gpt_data)

    all_ids = sorted(set(implicit_index.keys()) | set(gpt_index.keys()))
    diffs = []

    for item_id in all_ids:
        implicit_item = implicit_index.get(item_id)
        gpt_item = gpt_index.get(item_id)

        if implicit_item is None:
            diffs.append(
                {
                    "Id": item_id,
                    "Question": gpt_item["question"],
                    "implicit_intermediate_question": "",
                    "implicit_answer_type": "",
                    "gpt_intermediate_question": gpt_item["intermediate_question"],
                    "gpt_answer_type": gpt_item["answer_type"],
                    "difference_type": "missing_in_implicit_file",
                }
            )
            continue

        if gpt_item is None:
            diffs.append(
                {
                    "Id": item_id,
                    "Question": "",
                    "implicit_intermediate_question": implicit_item["intermediate_question"],
                    "implicit_answer_type": implicit_item["answer_type"],
                    "gpt_intermediate_question": "",
                    "gpt_answer_type": "",
                    "difference_type": "missing_in_gpt_file",
                }
            )
            continue

        question_diff = implicit_item["intermediate_question"] != gpt_item["intermediate_question"]
        answer_type_diff = implicit_item["answer_type"] != gpt_item["answer_type"]

        if question_diff or answer_type_diff:
            if question_diff and answer_type_diff:
                difference_type = "intermediate_question_and_answer_type_different"
            elif question_diff:
                difference_type = "intermediate_question_different"
            else:
                difference_type = "answer_type_different"

            diffs.append(
                {
                    "Id": item_id,
                    "Question": gpt_item["question"],
                    "implicit_intermediate_question": implicit_item["intermediate_question"],
                    "implicit_answer_type": implicit_item["answer_type"],
                    "gpt_intermediate_question": gpt_item["intermediate_question"],
                    "gpt_answer_type": gpt_item["answer_type"],
                    "difference_type": difference_type,
                }
            )

    return {
        "summary": {
            "implicit_file": str(IMPLICIT_PATH),
            "gpt_file": str(GPT_PATH),
            "implicit_count": len(implicit_data),
            "gpt_count": len(gpt_data),
            "diff_count": len(diffs),
        },
        "differences": diffs,
    }


def write_outputs(report):
    with OUTPUT_JSON.open("w", encoding="utf-8") as fp:
        json.dump(report, fp, ensure_ascii=False, indent=2)

    fieldnames = [
        "Id",
        "Question",
        "implicit_intermediate_question",
        "implicit_answer_type",
        "gpt_intermediate_question",
        "gpt_answer_type",
        "difference_type",
    ]

    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in report["differences"]:
            writer.writerow(row)


def main():
    if not IMPLICIT_PATH.exists():
        raise FileNotFoundError(f"File not found: {IMPLICIT_PATH}")
    if not GPT_PATH.exists():
        raise FileNotFoundError(f"File not found: {GPT_PATH}")

    report = compare_datasets()
    write_outputs(report)

    print("Comparison finished.")
    print(f"Diff count: {report['summary']['diff_count']}")
    print(f"JSON report: {OUTPUT_JSON}")
    print(f"CSV report: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
