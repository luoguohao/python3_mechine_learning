import pandas as pd
import numpy as np
import xgboost as xgb

data = pd.read_csv("seeds_dataset.txt", header=None,
                   sep=r'\s+', converters={7: lambda x: int(x) - 1})

data.rename(columns={7: 'label'}, inplace=True)

mask = np.random.rand(len(data)) > 0.8
train_data = data[mask]
test_data = data[~mask]

xgb_train = xgb.DMatrix(train_data.iloc[:, :6], label=train_data.label)
xgb_test = xgb.DMatrix(test_data.iloc[:, :6], label=test_data.label)

params = {
    # 'objective': 'multi:softmax',
    'objective': 'multi:softprob',
    'eta': 1,
    'max_depth': 5,
    'num_class': 3
}

watch_list = [(xgb_train, 'train'), (xgb_test, 'test')]
num_round = 50

bst = xgb.train(params, xgb_train, num_round, watch_list)

# dump 模型
bst.save_model('seed_train.model')
# 加载模型进行预测
bst1 = xgb.Booster()
bst1.load_model('seed_train.model')
pred = bst1.predict(xgb_test)
print(pred)

# 输出文本格式的模型（未做特征名称转化）
dump_model = bst.dump_model('seed_train_raw_model.txt')
# 输出文本格式的模型（完成特征名称转化）
dump_model1 = bst.dump_model('seed_train_nice_model.txt', 'featmap.txt')

# 取向量中预测值最大的分类作为y预测类型
pred_label = np.argmax(pred, axis=1)
print(pred_label)

error_rate = np.sum(pred_label != test_data.label) / test_data.shape[0]
print('测试集错误率（softmax）：{}'.format(error_rate))


if __name__ == '__main__':
    pass

