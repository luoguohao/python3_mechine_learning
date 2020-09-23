from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn import neighbors


def get_accuracy(testSet, predictions):
    """
    校验模型预测的准确性
    :param testSet:
    :param predictions:
    :return:
    """
    correct = 0
    # 遍历每个测试样本，判断是否预测正确并进行统计
    for i in range(len(testSet)):
        if testSet[i] == predictions[i]:
            correct += 1
    # 计算并返回准确性
    return (correct / float(len(testSet))) * 100.0


iris_data = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', sep=',',
                        names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])

print(iris_data.head(10))

# 柱状图显示各种花对应的属性特点
grouped_data = iris_data.groupby('class')
grouped_mean_data = grouped_data.mean()

grouped_mean_data.plot(kind='bar')
plt.legend(loc='best')
# plt.show()

# 采用留出法（hold-out)将样本分为训练集和测试集
msk = np.random.random(len(iris_data)) < 0.8
train_data = iris_data[msk]
test_data = iris_data[~msk]

print(len(train_data))
print(len(test_data))

# 重置索引
train_data = train_data.reset_index(drop=True)
test_data = test_data.reset_index(drop=True)

# 训练集label、测试集label
train_label = train_data['class']
test_label = test_data['class']

# 训练集特征、测试集特征
train_features = train_data.drop('class', axis=1)
test_features = test_data.drop('class', axis=1)

# 将特征归一化(min-max归一化方法）
train_features_norm = (train_features - train_features.min()) / (train_features.max() - train_features.min())
test_features_norm = (test_features - test_features.min()) / (test_features.max() - test_features.min())

knn = neighbors.KNeighborsClassifier(n_neighbors=3)
knn.fit(train_features, train_label)

predicts = knn.predict(test_features)
acc = get_accuracy(test_label, predicts)
print(acc)

if __name__ == '__main__':
    pass
