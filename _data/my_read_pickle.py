import pickle

# 读取pickle文件
with open(r'wikipedia_mappings.pickle', 'rb') as file:
    wikipedia_mappings = pickle.load(file)

# 输出前3个数据
# print(wikipedia_mappings[:3])

# print(type(wikipedia_mappings))#<class 'dict'> 这是一个字典，存储了ID和实体名称的键值对
# 获取字典的前3个键值对
print(list(wikipedia_mappings.items())[:3])#[('Q83803099', 'David_Balp'), ('Q83803433', 'Skype_a_Scientist'), ('Q83804735', 'Rae_Frances')]
