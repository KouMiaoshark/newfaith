import json
                                     # 需要注意relation的大小写区分
# function='train'
function='dev'

# output_path=f"merge_rela_{function}_solution1.json"

# input_path1=f"merge_rela_{function}_solution1.json"
input_path1=f"328dep_relation_dev_all.json"
input_path2=f"annotated_{function}_faith.json"

output_path=f"328依存句法树_82正确率_dev_merged.json"

# 1. 读取新的 relation
with open(input_path1, "r", encoding="utf-8") as f:
    new_Relation = json.load(f)

# 2. 读取原始 annotated 数据
with open(input_path2, "r", encoding="utf-8") as f:
    data = json.load(f)

# 如果你确实只想用前 6000 条
# data = data[:2000]

# 3. 建立 Id -> data_item 的映射
id_to_data = {}
for item in data:
    if "Id" in item:
        id_to_data[item["Id"]] = item
    else:
        print(f"警告：data 中存在无 Id 的条目，已跳过：{item}")

# 4. 用 new_Relation 覆盖 silver_tsf.relation
for item in new_Relation:
    try:
        item_id = item["Id"]
        # relation = item["Relation"]  #原有的提示词是Relation,   新的句法树提取是relation  大小写区分
        relation = item["relation"]


        if item_id not in id_to_data:
            # 这里会自然跳过 ID=2525
            print(f"警告：ID {item_id} 在 data 中不存在，跳过")
            continue

        current_data_item = id_to_data[item_id]

        # 确保 silver_tsf 存在
        if "silver_tsf" not in current_data_item or current_data_item["silver_tsf"] is None:
            print(f"警告：ID {item_id} 缺少 silver_tsf，跳过")
            continue

        if not isinstance(current_data_item["silver_tsf"], dict):
            print(f"警告：ID {item_id} 的 silver_tsf 不是 dict，跳过")
            continue

        # ✅ 正式替换 relation
        current_data_item["silver_tsf"]["relation"] = relation

    except KeyError as e:
        print(f"错误：缺少关键字段 {e}，item = {item}")
    except Exception as e:
        print(f"未知错误：{e}，item = {item}")

# 5. 保存新文件
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print(f"✅ relation 替换完成，结果已保存至 {output_path}")
