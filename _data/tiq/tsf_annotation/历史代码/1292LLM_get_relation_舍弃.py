import os
import json
from openai import OpenAI

# 请确保已安装 OpenAI SDK: `pip3 install openai`

client = OpenAI(
    api_key="sk-b34792bff4c84b44a76039f91c7c122a",
    base_url="https://api.deepseek.com"
)

# 读取 simplified_train.json 文件
with open('../simplified_train.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 创建保存简化后关系的列表
simplified_relations = []

# 每次处理 10 个问题
batch_size = 10
# for i in range(0, len(data), batch_size):
for i in range(0, 5000, batch_size):
    batch = data[i:i + batch_size]

    # 构建 messages
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",
         "content": "请根据下面提供的示例，提取每个问题中的 relation 字段。relation 应该表示最核心的实体间的关系，并且避免包含冗余的时间、地点等信息。请遵循以下规则：\n\n1. relation 应该简洁、清晰，聚焦于表达实体之间的主要关系。\n2. 任何描述时间、地点或背景的冗余部分应被去除，保持核心的动词或关系描述。\n3. 如果有时间约束，应该通过 Temporal Signal 字段来表达，而不是包含在 relation 字段中。\n4. 仅保留最简洁、直接的实体关系，避免长句子和复杂的背景描述。\n\n示例输出： {'Id': 1, 'Relation': 'was the Governor of'}"}
    ]

    # 将批次中的问题加入到 messages 中
    for item in batch:
        messages.append(
            {"role": "user", "content": json.dumps({"Id": item["Id"], "Question": item["Question"]})}
        )

    # 调用 Deepseek API 获取简化后的关系
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        stream=False
    )

    print(response)
    # 解析 API 输出并保存简化后的 relation
#     for i, item in enumerate(batch):
#         simplified_relation = response.choices[i].message.content.strip()  # 提取简化后的 relation
#         item["Relation"] = simplified_relation  # 更新关系
#
#     # 将结果添加到列表
#     simplified_relations.extend(batch)
#
# # 保存简化后的结果到 simplify_rela_output10.json 文件
# with open('simplify_rela_output10.json', 'w', encoding='utf-8') as f:
#     json.dump(simplified_relations, f, ensure_ascii=False, indent=4)
#
# print("Simplified relations have been saved to 'simplify_rela_output10.json'.")
    # 获取返回的内容
    response_content = response.choices[0].message.content.strip()
    if response_content.startswith("```json"):
        response_content = response_content[7:]  # 去掉 ```json
    if response_content.endswith("```"):
        response_content = response_content[:-3]  # 去掉末尾的 ```

    # 处理多个 JSON 对象
    relation_data = []
    for line in response_content.split("\n"):
        try:
            relation_item = json.loads(line)
            # 确保 'relation' 键存在
            if "Relation" in relation_item:
                relation_data.append(relation_item)
            else:
                print(f"Warning: 'Relation' key not found in item: {relation_item}")
        except json.JSONDecodeError:
            print(f"Error decoding line: {line}")

    # 将简化后的 relation 更新到数据中
    for relation_item in relation_data:
        for item in batch:
            if item["Id"] == relation_item["Id"]:
                item["Relation"] = relation_item["Relation"] #这里可以保留原本不好的。
                # item["relation"] = relation_item["relation"]

    # 将结果添加到列表
    simplified_relations.extend(batch)

# 保存简化后的结果到 simplify_rela_output10.json 文件
with open('simplify_rela_output10.json', 'w', encoding='utf-8') as f:
    json.dump(simplified_relations, f, ensure_ascii=False, indent=4)

print("Simplified relations have been saved to 'simplify_rela_output10.json'.")
