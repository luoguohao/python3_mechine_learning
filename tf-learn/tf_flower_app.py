import tensorflow as tf
import pathlib
import matplotlib.pyplot as plt


def load_and_preprocess_image(path):
    # 读取图片
    image = tf.io.read_file(path)
    # 将jpg格式图片解码，得到一个张量（三维矩阵）
    image = tf.image.decode_jpeg(image, channels=3)
    # 由于数据集中每张图片的大小不一样，所以统一调整为192*192
    image = tf.image.resize(image, [192, 192])
    # 对每个像素点的RGB值做归一化处理
    print(type(image))
    image /= 255.0

    return image


def change_range(image, label):
    """
    将范围在[0, 1]之间的数据映射到[-1, 1]之间
    :param image:
    :param label:
    :return:
    """
    return 2 * image - 1, label


# 获取当前路径
data_root = pathlib.Path('/Users/didi/Documents/didi_workspace/datasets/flower_photos')
# 获取指定目录下的文件路径(返回的是一个列表，每一个元素是一个PosixPath对象）
all_image_paths = list(data_root.glob('*/*'))
print(type(all_image_paths[0]))

# 将PosixPath对象转为字符串
all_image_paths = [str(path) for path in all_image_paths]

# 获取图片类别名字，（存放样本图片的五个文件夹名称）
label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
# 将类别名称转换成数值类型的类标
label_to_index = dict((name, index) for index, name in enumerate(label_names))
# 获取所有图片的类标
all_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in all_image_paths]

print(label_to_index)
print("First 2 labels indices:", all_image_labels[:2])
print("First 2 labels paths:", all_image_paths[:2])

# 构建图片路径的数据集
path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
# 使用AUTOTUNE自动调节管道参数
AUTOTUNE = tf.data.experimental.AUTOTUNE
# 构建图片数据的数据集
image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
# 构建类标数据的数据集
label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))
# 将图片和类压缩为（图片，类标）对
image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))

print(image_ds)
print(label_ds)
print(image_label_ds)

# 数据集中部分数据可视化
plt.figure(figsize=(8, 8))
for n, image_label in enumerate(image_label_ds.take(4)):
    print(image_label[0])
    print(image_label[1])
    plt.subplot(2, 2, n + 1)
    plt.imshow(image_label[0])
    plt.grid(False)
    plt.xticks([])  # 去除x轴坐标点
    plt.yticks([])  # 去除y轴坐标点
    plt.xlabel(str(image_label[1]))  # 设置x轴标签值为image_label的tensor描述

# plt.show()

# 简单使用训练好的MobileNetV2模型，将其迁移到花朵分类任务上
# 默认下载的模型在用户根目录下，具体位置是"～/.keras/models/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_192_no_top.h5
mobile_net = tf.keras.applications.MobileNetV2(input_shape=(192, 192, 3), include_top=False)
# 禁止训练更新"MobileNetV2"模型参数
mobile_net.trainable = False

# 打乱数据集，定义训练批次大小
image_count = len(all_image_paths)
ds = image_label_ds.shuffle(buffer_size=image_count)
# 让数据集重复多次(不指定重复次数，表示无限次重复)
ds = ds.repeat()
# 设置每个批次的大小
BATCH_SIZE = 32
ds = ds.batch(BATCH_SIZE)
# 通过"prefetch"方法让模型的训练和每个批次数据加载并行
ds = ds.prefetch(buffer_size=AUTOTUNE)

# 由于MobileNetV2接受的输入数据是归一化在[-1, 1]之间的数据，而图片的归一化之后，其范围为[0,1]，需要重新映射一下
# 使用map方法对数据集进行处理
keras_ds = ds.map(change_range)
# 由于MobileNetV2返回的数据维度是(32,6,6,1280),其中32是一个批次数据大小，6*6表示特征图大小为6*6，1280表示该层使用了
# 1280个卷机核，为了适应花朵分类任务，需要在该模型返回数据的基础上再增加两层网络层
model = tf.keras.Sequential([mobile_net,
                             tf.keras.layers.GlobalAveragePooling2D(),
                             tf.keras.layers.Dense(len(label_names))
                             ])
# 全局平均池化（Global Average Pooling, GAP）是对每一个特征图池化后的结果，将该平均值作为特征图池化后的结果，因此经过该操作之后
# 数据的维度变成了（32，1280），由于花朵任务是一个5分类任务，因此需要一个全连接(Dense)将维度变为（32，5）

# 编译模型
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# 输出模型各层的参数概况
model.summary()

# 模型训练,epochs表示训练的回合数，steps_per_epoch表示每个回合要取多少个批次数，通常steps_per_epoch大小等于数据集大小除以批次大小后上取整
model.fit(ds, epochs=3, steps_per_epoch=1000)

if __name__ == '__main__':
    pass
