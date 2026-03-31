import json
import os
import random

# =========================
# 手动修改这几个参数即可
# =========================
sample_size = 50
random_seed = 42   # 想固定抽样结果就填数字；不想固定就改成 None

INPUT_PATH = "_benchmarks/tiq/test.json"       
OUTPUT_PATH = f"_benchmarks/tiq/test_random_{sample_size}.json"


def sample_json_dataset(input_path, output_path, sample_size, random_seed=None):
    # 检查输入文件是否存在
    if not os.path.exists(input_path):
        print(f"错误：输入文件不存在 -> {input_path}")
        return

    # 读取原始 JSON
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 检查数据格式
    if not isinstance(data, list):
        print("错误：JSON 顶层不是列表，无法按条目抽样。")
        return

    # 检查抽样数量是否合法
    if sample_size > len(data):
        print(f"错误：抽样数量 {sample_size} 大于数据总条数 {len(data)}。")
        return

    # 固定随机种子（可选）
    if random_seed is not None:
        random.seed(random_seed)

    # 随机抽取 sample_size 条
    subset = random.sample(data, sample_size)

    # 保存到新文件
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(subset, f, ensure_ascii=False, indent=4)

    print("处理完成。")
    print(f"原始文件: {input_path}")
    print(f"原始条数: {len(data)}")
    print(f"随机抽取条数: {len(subset)}")
    print(f"随机种子: {random_seed}")
    print(f"输出文件: {output_path}")


if __name__ == "__main__":
    sample_json_dataset(INPUT_PATH, OUTPUT_PATH, sample_size, random_seed)