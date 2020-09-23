import tensorflow as tf
from tensorflow.python.client import device_lib

import os
# 选择编号为0的GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def get_available_gpus():
    local_devices_protos = device_lib.list_local_devices()
    return [x.name for x in local_devices_protos if x.device_type == 'gpu']


if __name__ == '__main__':
    print(get_available_gpus())
