import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

model = tf.keras.Sequential([
    # 添加一个64个神经元的全连接层，"input_shape"为该层接受的输入数据的维度，"activation"指定该层所用的激活函数
    layers.Dense(64, activation='relu', input_shape=(32,)),
    # 添加第二个网络层
    layers.Dense(64, activation='relu'),
    # 添加一个softmax层作为输出层，该层有十个单元
    layers.Dense(10, activation='softmax')
])

# 同样可以通过add方法增加网络层
model1 = tf.keras.Sequential()
model1.add(layers.Dense(64, activation='relu', input_shape=(32,)))
model1.add(layers.Dense(64, activation='relu'))
model1.add(layers.Dense(10, activation='softmax'))

# 创建好网络后，对网络编译: optimizers用来指定使用的优化器及优化器的学习率，如Adam优化器：tf.keras.optimizer.Adam，
# SGD优化器:tf.keras.optimizer.SGD
model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# 使用一组随机数作为训练数据
data = np.random.random((1000, 32))
labels = np.random.random((1000, 10))
print(data[0])
print(labels[0])

# 在训练模型中，为了更好的调节参数，方便模型选择和优化，通常会准备一个验证集
val_data = np.random.random((100, 32))
val_labels = np.random.random((100, 10))
model.fit(data, labels, epochs=2, batch_size=32, validation_data=(val_data, val_labels))
# 使用验证集对模型进行评估
print(model.evaluate(data, labels, batch_size=50))

# 同样可以使用tf.data将numpy数据转为DataSet数据集，再传递给模型训练
dataset = tf.data.Dataset.from_tensor_slices((data, labels))
dataset = dataset.batch(50)
# 验证集
val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_labels))
val_dataset = val_dataset.batch(50)
model.fit(dataset, epochs=2, validation_data=val_dataset)
# 使用验证集对模型进行评估
print(model.evaluate(dataset, batch_size=50))   # loss & accuracy
# 预测
result = model.predict(np.random.random((100, 32)), batch_size=40)
print(result.argmax(axis=1))

## 函数式api构建复杂的网络层
# 单独一个输入层
inputs = tf.keras.Input(shape=(32,))
# 网络层可以像函数一样调用，其接收和输出都为张量
x = layers.Dense(64, activation='relu')(inputs)
x = layers.Dense(64, activation='relu')(x)
# 输出层
predictions = layers.Dense(10, activation='softmax')(x)
# 创建模型
model = tf.keras.Model(inputs=inputs, outputs=predictions)
# 编译模型
model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001), loss='categorical_crossentropy', metrics=['accuracy'])
# 训练模型
model.fit(data, labels, epochs=2, batch_size=50)


### 自定义模型类和网络层
class MyModel(tf.keras.Model):

    def __init__(self, num_classes=10):
        super().__init__(name='my_model')
        # 分类任务的类别数
        self.num_classes = num_classes
        # 定义自己的网络层
        self.dense_1 = layers.Dense(32, activation='relu')
        self.dense_2 = layers.Dense(num_classes, activation='sigmoid')

    def call(self, inputs, training=None, mask=None):
        # 使用__init__方法中定义的网络层来构造网络的前馈过程
        x = self.dense_1(inputs)
        return self.dense_2(x)


# 使用自定义模型
model = MyModel(num_classes=10)
# 编译模型
model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001), loss='categorical_crossentropy', metrics=['accuracy'])
# 训练模型
model.fit(data, labels, epochs=2, batch_size=50)

callbacks = [
    # 若验证集上的损失'val_loss'连续两个训练回合没有变化，则提前结束训练
    tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
    # 使用TensorBoard把训练记录保存到'./logs'目录中
    tf.keras.callbacks.TensorBoard(log_dir='./logs')
]

model.fit(data, labels, epochs=2, batch_size=50, validation_data=(val_data, val_labels), callbacks=callbacks)

# 将模型保存为HDF5文件,通过save()方法保存的是一个完成模型信息，包含模型的权重和结构
model.save('my_model')
# 加载模型
model = tf.keras.models.load_model('my_model')

# 可以单独保存模型的权重参数或者模型的结构
model.save_weights('my_model_weight.h5', save_format='h5')
# 重新加载
model.load_weights('my_model_weight.h5')
# 将模型结构保存为json
json_str = model1.to_json()
print(json_str)

strategy = tf.distribute.MirroredStrategy()
# 优化器及模型的构建、编译必须在scope()中
with strategy.scope():
    # 同样可以通过add方法增加网络层
    model2 = tf.keras.Sequential()
    model2.add(layers.Dense(64, activation='relu', input_shape=(32,)))
    model2.add(layers.Dense(64, activation='relu'))
    model2.add(layers.Dense(10, activation='softmax'))
    model2.compile(optimizer=tf.keras.optimizers.RMSprop(0.001), loss='categorical_crossentropy', metrics=['accuracy'])

if __name__ == '__main__':
    pass
