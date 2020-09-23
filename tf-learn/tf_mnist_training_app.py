import numpy as np
import struct
import tensorflow as tf


class MnistData:
    def __init__(self, train_image_path, train_label_path, test_image_path, test_label_path):
        # 训练集和测试集的文件路径
        self.train_image_path = train_image_path
        self.train_label_path = train_label_path
        self.test_image_path = test_image_path
        self.test_label_path = test_label_path
        # 获取验证集和训练集数据
        self.train_images, self.train_labels = self.get_data(0)
        self.test_images, self.test_labels = self.get_data(1)

    def get_data(self, data_type):
        '''
            由于MNIST数据是以二进制文件的形式存储的，所以需要用到struct模块来处理文件，uppack_from函数用来解包二进制文件，
            参数“>IIII”指定读取16个字节的内容，这正好是文件的基本信息部分。其中“>”代表二进制文件是以大端法存储的，
            “IIII”代表四个int类型的长度，这里一个int类型占4个字节。参数“image_file”是要读取的文件，“image_index”是偏置量。
            如果要连续地读取文件内容，每读取一部分数据后就要增加相应的偏置量
        '''
        if data_type == 0:  # 获取训练集数据
            image_path = self.train_image_path
            label_path = self.train_label_path
        else:
            image_path = self.test_image_path
            label_path = self.test_label_path
        
        with open(image_path, 'rb') as file1:
            image_file = file1.read()
        with open(label_path, 'rb') as file2:
            label_file = file2.read()
        
        label_index = 0
        image_index = 0
        labels = []
        images = []

        # 读取训练集图片护具文件的文件信息
        _, num_of_datasets, _, _ = struct.unpack_from('>IIII', image_file, image_index)
        image_index += struct.calcsize('>IIII')

        for i in range(num_of_datasets):
            # 读取784个unsigned byte,即一幅图片的所有像素值
            temp = struct.unpack_from('>784B', image_file, image_index)
            # 将读取的像素转成28*28的矩阵
            temp = np.reshape(temp, (28, 28))
            # 归一化处理
            temp = temp / 255
            images.append(temp)
            image_index += struct.calcsize('>784B') # 每次增加784B

        # 跳过描述信息
        label_index += struct.calcsize('>II')
        labels = struct.unpack_from('>' + str(num_of_datasets) + 'B', label_file, label_index)
        # One-hot编码
        labels = np.eye(10)[np.array(labels)]
        return np.array(images), labels


if __name__ == "__main__":
    mnist_data = MnistData(
        '/Users/didi/Documents/didi_workspace/datasets/mnist/train-images-idx3-ubyte',
        '/Users/didi/Documents/didi_workspace/datasets/mnist/train-labels-idx1-ubyte',
        '/Users/didi/Documents/didi_workspace/datasets/mnist/t10k-images-idx3-ubyte',
        '/Users/didi/Documents/didi_workspace/datasets/mnist/t10k-labels-idx1-ubyte'
        ) 
    print(mnist_data.train_labels)

    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28,28)),
            tf.keras.layers.Dense(512, activation=tf.nn.relu),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation=tf.nn.softmax)
        ]
    )

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    # 模型训练
    model.fit(mnist_data.train_images, mnist_data.train_labels, epochs=10)
    # 模型验证
    model.evaluate(mnist_data.test_images, mnist_data.test_labels)