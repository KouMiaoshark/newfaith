import json
import os

# =========================
# 手动修改这几个参数即可
# =========================
start_id = 100
end_id   = 200     # 截取前多少条

INPUT_PATH = "_benchmarks/tiq/test.json"       # 原始数据集文件
OUTPUT_PATH = f"_benchmarks/tiq/test_{start_id}_{end_id}.json"   # 裁剪后输出文件
                         

def cut_json_dataset(input_path, output_path, start_id,end_id):
    # 检查输入文件是否存在
    if not os.path.exists(input_path):
        print(f"错误：输入文件不存在 -> {input_path}")
        return

    # 读取原始 JSON
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 检查数据格式
    if not isinstance(data, list):
        print("错误：JSON 顶层不是列表，无法按条目裁剪。")
        return

    # 截取 指定范围 条
    subset = data[start_id:end_id]

    # 保存到新文件
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(subset, f, ensure_ascii=False, indent=4)

    print("处理完成。")
    print(f"原始文件: {input_path}")
    print(f"原始条数: {len(data)}")
    print(f"截取条数: {len(subset)}，从{start_id}到{end_id}")
    print(f"输出文件: {output_path}")

if __name__ == "__main__":
    cut_json_dataset(INPUT_PATH, OUTPUT_PATH, start_id,end_id)