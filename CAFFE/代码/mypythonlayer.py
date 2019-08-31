# -*- coding: utf-8 -*-
import sys

caffe_root = '/home/bertram-liu/caffe/'
sys.path.insert(0, caffe_root + 'python')

import caffe
import numpy as np
import yaml  # 参数传递
import cv2  # opencv-python接口

# 自定义MyLayer层
class MyLayer(caffe.Layer):  # 对应conv.prototxt文件

    # 准备工作
    def setup(self, bottom, top):
        self.num = yaml.load(self.param_str)["num"]
        print("Parameter num : ", self.num)

    def reshape(self, bottom, top):
        pass

    # 前向传播
    def forward(self, bottom, top):
        top[0].reshape(*bottom[0].shape)  # 第0个输出, 第0个输入
        print(bottom[0].data.shape)
        print(bottom[0].data)

        top[0].data[...] = bottom[0].data + self.num  # 对应conv.prototxt: 21
        print(top[0].data[...])

    # 反向传播
    def backward(self, top, propagate_down, bottom):
        pass


net = caffe.Net('conv.prototxt', caffe.TEST)
# im = np.array(Image.open('timg.jpeg'))

# 加载图片
im = np.array(cv2.imread('timg.jpeg'))
print(im.shape)  # 三维的
# im_input = im[np.newaxis, np.newaxis, :, :]

# 添加一个新的维度  都是4维的 b x c x h x w
im_input = im[np.newaxis, :, :]
print(im_input.shape)

# print(im_input.transpose((1,0,2,3)).shape)

# 变换 (1, 496, 700, 3) >>> (1, 3, 496, 700)
im_input2 = im_input.transpose((0, 3, 1, 2))
print(im_input2.shape)
# print(im_input.shape)

# 把配置文件[100, 100] >>> [496, 700]
net.blobs['data'].reshape(*im_input2.shape)
net.blobs['data'].data[...] = im_input2
net.forward()
