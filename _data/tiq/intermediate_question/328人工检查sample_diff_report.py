import json
import random
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent
INPUT_PATH = CURRENT_DIR / "intermediate_question_diff_report.json"
OUTPUT_PATH = CURRENT_DIR / "intermediate_question_diff_report_sample_50人工检查.json"
SAMPLE_SIZE = 50
RANDOM_SEED = 42


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"File not found: {INPUT_PATH}")

    with INPUT_PATH.open("r", encoding="utf-8") as fp:
        report = json.load(fp)

    differences = report.get("differences", [])
    if not isinstance(differences, list):
        raise ValueError("The 'differences' field is not a list.")

    sample_size = min(SAMPLE_SIZE, len(differences))
    rng = random.Random(RANDOM_SEED)
    sampled = rng.sample(differences, sample_size)

    output = {
        "summary": {
            "source_file": str(INPUT_PATH),
            "original_diff_count": len(differences),
            "sample_size": sample_size,
            "random_seed": RANDOM_SEED,
        },
        "differences": sampled,
    }

    with OUTPUT_PATH.open("w", encoding="utf-8") as fp:
        json.dump(output, fp, ensure_ascii=False, indent=2)

    print("Sampling finished.")
    print(f"Original diff count: {len(differences)}")
    print(f"Sample size: {sample_size}")
    print(f"Output file: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
