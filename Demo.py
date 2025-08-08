import cv2
import numpy as np
import torch
from utils.processing import resize_with_aspect_ratio
from utils.net import ResNet18 as Model
import os 



labels = ['blue', 'red']
for file in os.listdir(r"C:\Users\hhkj\Desktop\images\blue"):
    file_path = os.path.join(r"C:\Users\hhkj\Desktop\images\blue", file)
    img = cv2.imread(file_path)
    img = resize_with_aspect_ratio(img, (224,224))
    img = img/255.0  # 归一化到0-1之间
    # img = np.transpose(img, (2, 0, 1))  # 转换为CHW
    img = np.expand_dims(img, axis=0)  # 增加一维
    img = torch.from_numpy(img).float()
    model = Model()
    model.load_state_dict(torch.load(r"E:\DMH\Shanghai\dmh_categorization\train\runs\exp6\weights\best.pt"))
    model.eval()
    with torch.no_grad():
        output = model(img)
        result = torch.argmax(output, dim=1)
        print(labels[result])
