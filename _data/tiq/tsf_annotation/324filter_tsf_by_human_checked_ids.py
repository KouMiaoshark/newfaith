import argparse
import json
from pathlib import Path


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def filter_by_id(source_data, checked_data, id_key="Id"):
    checked_ids = {item[id_key] for item in checked_data if id_key in item}
    filtered_data = [item for item in source_data if item.get(id_key) not in checked_ids]
    return filtered_data, checked_ids


def main():
    parser = argparse.ArgumentParser(
        description="Remove items from a source JSON list when their Id appears in a checked JSON list."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("324移除句法分析LLM_训练集.json"),  #被移除id的文件
        help="Path to the source dataset JSON file.",
    )
    parser.add_argument(
        "--checked",
        type=Path,
        default=Path("322_50_2_human_check.json"),
        help="Path to the human-checked dataset JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("324移除句法分析LLM_训练集_移除人工标注问题_附带显性问题.json"),
        help="Path to the filtered output JSON file.",
    )
    parser.add_argument(
        "--id-key",
        default="Id",
        help="Key name used as the unique identifier.",
    )
    args = parser.parse_args()

    source_data = load_json(args.source)
    checked_data = load_json(args.checked)

    if not isinstance(source_data, list) or not isinstance(checked_data, list):
        raise ValueError("Both input files must contain a JSON list.")

    filtered_data, checked_ids = filter_by_id(source_data, checked_data, args.id_key)
    removed_count = len(source_data) - len(filtered_data)

    dump_json(args.output, filtered_data)

    print(f"Source items: {len(source_data)}")
    print(f"Checked ids: {len(checked_ids)}")
    print(f"Removed items: {removed_count}")
    print(f"Remaining items: {len(filtered_data)}")
    print(f"Output written to: {args.output}")


if __name__ == "__main__":
    main()
