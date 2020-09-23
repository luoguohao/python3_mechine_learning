import json

import xgboost as xgb

xgb_train = xgb.DMatrix('12.agaricus_train.txt')
xgb_test = xgb.DMatrix('12.agaricus_test.txt')

# 定义模型参数
params = {
    "objective": "binary:logistic",
    "booster": "gbtree",
    "max_depth": 3
}

# 训练轮数
num_round = 5

# 训练过程中实时输出评估结果
watchlist = [(xgb_train, 'train'), (xgb_test, 'test')]

# 模型训练
model = xgb.train(params, xgb_train, num_round, watchlist)

# 模型预测
predicts = model.predict(xgb_test)

print(predicts)

if __name__ == '__main__':
    pass
