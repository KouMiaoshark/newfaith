import json
import random
from pathlib import Path

INPUT_FILE = "328dep_relation_train_all.json"
OUTPUT_FILE = "328_50_依存句法树human_check.json"
SEED = 42  # 固定随机种子，便于复现；不需要可改成 None

def main():
    input_path = Path(INPUT_FILE)
    if not input_path.exists():
        raise FileNotFoundError(f"未找到输入文件: {input_path.resolve()}")

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("输入 JSON 顶层必须是列表。")

    first_6000 = data[:6000]
    if len(first_6000) < 50:
        raise ValueError(f"前6000条中实际只有 {len(first_6000)} 条，无法抽取50条。")

    if SEED is not None:
        random.seed(SEED)

    sampled = random.sample(first_6000, 50)

    output_path = Path(OUTPUT_FILE)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(sampled, f, ensure_ascii=False, indent=4)

    print(f"已从前{len(first_6000)}条中随机抽取50条，保存到: {output_path.resolve()}")

if __name__ == "__main__":
    main()
