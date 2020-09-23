import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
import pandas as pd


def create_more_data(images):
    '''
        虽然旋转前后的图片是同一张，但是由于特征的位置发生了变化，因此对于模型来说就是不同的数据，旋转或翻转操作是扩充图像数据集的一个简单有效的方法。
        对应的操作分别是逆时针旋转90°、逆时针旋转180°、逆时针旋转270°、左右翻转和上下翻
    '''
    # 通过旋转、翻转扩充数据
    image_rot90 = []
    image_rot180 = []
    image_rot270 = []
    image_lr = []
    image_ud = []

    for i in range(0, images.shape[0]):
        band_1 = images[i,:,:,0]
        band_2 = images[i,:,:,1]
        band_3 = images[i,:,:,2]

        # 旋转90度
        band_1_rot90 = np.rot90(band_1)
        band_2_rot90 = np.rot90(band_2)
        band_3_rot90 = np.rot90(band_3)
        image_rot90.append(np.dstack((band_1_rot90, band_2_rot90, band_3_rot90)))

        # 旋转180度
        band_1_rot180 = np.rot90(band_1, 2)
        band_2_rot180 = np.rot90(band_2, 2)
        band_3_rot180 = np.rot90(band_3, 2)
        image_rot180.append(np.dstack((band_1_rot180, band_2_rot180, band_3_rot180)))

        # 旋转270度
        band_1_rot270 = np.rot90(band_1, 3)
        band_2_rot270 = np.rot90(band_2, 3)
        band_3_rot270 = np.rot90(band_3, 3)
        image_rot270.append(np.dstack((band_1_rot270, band_2_rot270, band_3_rot270)))

        # 左右翻转
        lr1 = np.flip(band_1, 0)
        lr2 = np.flip(band_2, 0)
        lr3 = np.flip(band_3, 0)
        image_lr.append(np.dstack((lr1, lr2, lr3)))
        
        # 上下翻转
        ud1 = np.flip(band_1, 1)
        ud2 = np.flip(band_2, 1)
        ud3 = np.flip(band_3, 1)
        image_ud.append(np.dstack((ud1, ud2, ud3)))

    rot90 = np.array(image_rot90)
    rot180 = np.array(image_rot180)
    rot270 = np.array(image_rot270)
    rotlr = np.array(image_lr)
    rotud = np.array(image_ud)
    images = np.concatenate((images, rot90, rot180, rot270, rotlr, rotud))
    return images
        

def data_process(path, more_data): 
    # 读取数据
    data_frame = pd.read_json(path)
    # 获取图像数据
    images = []
    for _, rows in data_frame.iterrows():
        # 将一维数据转换为75*75的二维数据
        band_1 = np.array(rows['band_1']).reshape(75, 75)
        band_2 = np.array(rows['band_2']).reshape(75, 75) 
        band_3 = band_1 + band_2
        # 除了原有的“band_1”和“band_2”，我们增加了“band_3”，band_3=band_1+band_2。
        # 最后使用NumPy的“dstack”函数将三种数据进行堆叠，因此我们单个样本的数据维度为75×75×3。
        images_i = np.dstack((band_1, band_2, band_3))
        images.append(images_i)

    if more_data:
        # 扩充数据
        images = create_more_data(np.array(images))
    
    # 获取类标
    labels = np.array(data_frame['is_iceberg'])
    if more_data:
        # 扩充数据后，类标也需要相应扩充,因为“create_more_data”函数将训练数据扩充为了原来的6倍，所以这里也要对应地将类标扩充为原来的6倍。
        labels = np.concatenate((labels,labels,labels,labels,labels,labels))

    return np.array(images), labels
    

def get_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(128, kernel_size=(3,3), activation='relu', input_shape=(75, 75, 3)))
    model.add(layers.MaxPool2D(pool_size=(3,3),strides=(2,2)))
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(128, kernel_size=(3,3), activation='relu'))
    model.add(layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(layers.Dropout(0.2)) 

    model.add(layers.Conv2D(64, kernel_size=(2,2), activation='relu'))
    model.add(layers.MaxPool2D(pool_size=(3,3),strides=(2,2)))
    model.add(layers.Dropout(0.2)) 

    model.add(layers.Conv2D(64, kernel_size=(2,2), activation='relu'))
    model.add(layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(layers.Dropout(0.2)) 

    # 将上一层的输出特征映射转化为一维数据，以便进行全连接操作
    model.add(layers.Flatten())
    model.add(layers.Dense(256))
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.2))  # 在池化层的后面进行Dropout操作，丢弃了20%的神经元，防止参数过多导致过拟合

    model.add(layers.Dense(128))
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.2)) 

    model.add(layers.Dense(1))
    model.add(layers.Activation('sigmoid'))

    # 编译模型
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.00001), metrics='accuracy')
    model.summary()
    return model


if __name__ == "__main__":
    train_x, train_y = data_process('/Users/didi/Documents/didi_workspace/datasets/iceberg_datas/train.json', True)
    cnn_model = get_model()
    cnn_model.fit(train_x, train_y, batch_size=25, epochs=100, verbose=1, validation_split=0.2)