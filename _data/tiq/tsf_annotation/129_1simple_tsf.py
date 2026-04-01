import json


input_path="317test.json"
output_path="simplified_317test.json"
# 场景1：如果你已经有本地原始.json文件（推荐，假设文件名为 original_data.json）
# 步骤1：读取原始JSON文件
try:
    with open(input_path, "r", encoding="utf-8") as f:
        original_data = json.load(f)
except FileNotFoundError:
    print("未找到本地original_data.json文件，将使用提供的示例数据进行演示")
    # 场景2：直接使用你提供的JSON示例数据（补全了缺失的闭合符号，保证JSON格式合法）
    original_data = [
        {
            "Id": 1,
            "Question": "Who was the Autonomous Governorate of Estonia's Governor of Estonia when they declared sovereignty as the Estonian Autonoomne Eestimaa kubermang?",
            "Temporal signal": ["OVERLAP"],
            "Temporal question type": ["Implicit"],
            "Answer": [{"AnswerType": "Entity", "WikidataQid": "Q1385836", "WikidataLabel": "Jaan Poska", "WikipediaURL": "https://en.wikipedia.org/wiki/Jaan_Poska"}],
            "Data source": "TIQ",
            "Question creation date": "2023-07-15",
            "Data set": "train_url",
            "silver_tsf": {
                "entity": ["Autonomous Governorate of Estonia", "Estonia's", "estonia"],
                "relation": "Who was the Governor of when they declared sovereignty as the Estonian Autonoomne Eestimaa kubermang",
                "answer_type": "human",
                "temporal_signal": "OVERLAP",
                "categorization": "Implicit"
            },
            "question": "Who was the Autonomous Governorate of Estonia's Governor of Estonia when they declared sovereignty as the Estonian Autonoomne Eestimaa kubermang?",
            "answers": [{"id": "Q1385836", "label": "Jaan Poska"}]
        },
        {
            "Id": 2,
            "Question": "Who founded Bucks County when it was one of Pennsylvania's three original counties created by the colonial proprietor?",
            "Temporal signal": ["OVERLAP"],
            "Temporal question type": ["Implicit"],
            "Answer": [{"AnswerType": "Entity", "WikidataQid": "Q209152", "WikidataLabel": "William Penn", "WikipediaURL": "https://en.wikipedia.org/wiki/William_Penn"}],
            "Data source": "TIQ",
            "Question creation date": "2023-07-15",
            "Data set": "train_url",
            "silver_tsf": {
                "entity": ["Bucks County", "Pennsylvania"],
                "relation": "Who founded when it was one of three original counties created by the colonial proprietor",
                "answer_type": "human",
                "temporal_signal": "OVERLAP",
                "categorization": "Implicit"
            }
        }
    ]

# 步骤2：提取仅包含Id和silver_tsf的简化数据
simplified_data = []
for item in original_data:
    # 先获取silver_tsf，处理None情况
    silver_tsf_data = item.get("silver_tsf", {})
    # 确保silver_tsf_data是字典（如果是None，强制转为空字典）
    if not isinstance(silver_tsf_data, dict):
        silver_tsf_data = {}

    # 构建简化条目
    simplified_item = {
        "Id": item.get("Id", None),
        "Question": item.get("Question", None),
        # "entity": silver_tsf_data.get("entity", None),
        "relation": silver_tsf_data.get("relation", None),
        "origin_entity": silver_tsf_data.get("entity",None),
        "Answer":item.get("Answer", None)
    }
    simplified_data.append(simplified_item)


# 步骤3：将简化后的数据保存到新的JSON文件
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(simplified_data, f, ensure_ascii=False, indent=4)

print(f"简化完成！新文件已保存为 {output_path}")