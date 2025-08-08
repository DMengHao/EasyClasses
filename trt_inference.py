# -*- coding:utf-8 -*-
import time
import cv2
import numpy as np
import torch
import tensorrt as trt
from pycuda import driver
import pycuda.driver as cuda0
from collections import OrderedDict, namedtuple
from utils.processing import resize_with_aspect_ratio
import pycuda.autoinit

class trt_inference:
    def __init__(self, weights='./weights/new_data_20250222.engine', imgsz=224, dev='cuda:0'):
        self.imgsz = imgsz
        self.device = int(dev.split(':')[-1])
        # 创建一个CUDA上下文
        self.ctx = cuda0.Device(self.device).make_context()
        # 创建一个CUDA流，用于管理异步操作顺序。通过流可以将输入数据从CPU拷贝到GPU，Engine执行推理，将输出数据从GPU拷贝回CPU。这三个步骤按顺序异步执行，减少CPU等待GPU的时间。
        self.stream = driver.Stream()
       
        logger = trt.Logger(trt.Logger.INFO)
        with open(weights, 'rb') as f:
            self.runtime = trt.Runtime(logger)
            self.model = self.runtime.deserialize_cuda_engine(f.read())
        self.context = self.model.create_execution_context() # 创建执行上下文
        self.bindings = OrderedDict()
        Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        # 统一使用新API（不再需要版本判断）
        for index in range(self.model.num_bindings):
            # 获取张量名称（新API）
            name = self.model.get_tensor_name(index)
            # 获取数据类型（新API）
            dtype = trt.nptype(self.model.get_tensor_dtype(name))
            # 获取形状（新API）
            shape = tuple(self.model.get_tensor_shape(name))
            # 创建PyTorch张量
            data = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(self.device)
            # 存储绑定信息
            self.bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))
            del data  # 显式释放引用
        # 生成指针地址字典
        self.binding_addrs = OrderedDict((n, d.ptr) for n, d in self.bindings.items())


    def infer(self, tensor_data):
        try:
            self.ctx.push()
            self.binding_addrs['input'] = int(tensor_data.data_ptr())
            self.context.execute_v2(list(self.binding_addrs.values()))
            self.stream.synchronize()  # 确保所有计算已完成，阻塞调用
            preds = self.bindings['output'].data
            result = torch.argmax(preds, dim=1)
            return result
        finally:
            self.ctx.pop()


    def __del__(self):
        self.ctx.pop()
        del self.context
        del self.model
        del self.runtime


if __name__ == '__main__':
    labels = ['blue', 'red']
    trt_api = trt_inference(weights=r'E:\DMH\Shanghai\dmh_categorization\export\model_224_fp32.engine',imgsz=224, dev='cuda:0')
    for i in range(50):
        start_time = time.time()
        img = cv2.imread(r'C:\Users\hhkj\Desktop\images\red\person_20250630_011425_1.jpg')
        img = resize_with_aspect_ratio(img, (224,224))
        img = img/255.0  # 归一化到0-1之间
        img = np.expand_dims(img, axis=0)  # 增加一维
        img = torch.from_numpy(img).float().to('cuda:0')
        detections = trt_api.infer(img)
        print(labels[detections])
        print(f'处理用时：{(time.time() - start_time) * 1000:.4f}ms')