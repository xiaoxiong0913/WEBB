import pickle

# 加载模型文件的本地路径
model_path = "C:\\Users\\14701\\Desktop\\WEBB\\treebag_model.pkl"

# 加载模型
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

# 检查模型的特征名称（如果可用）
try:
    feature_names_in_ = model.feature_names_in_
    print("模型期望的特征名称：", feature_names_in_)
except AttributeError:
    print("模型没有保存特征名称信息。")

# 打印模型的参数
print("模型参数：", model.get_params())
