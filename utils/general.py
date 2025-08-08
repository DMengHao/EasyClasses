import re
import glob
from pathlib import Path
import os
import torch
from matplotlib import pyplot as plt
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def increment_path(path, exist_ok=False, sep='', mkdir=True):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        path = Path(f"{path}{sep}{n}{suffix}")  # increment path
    if mkdir:
        os.makedirs(os.path.join(path, 'weights'), exist_ok=True)  
        path.mkdir(parents=True, exist_ok=True)  # make directory
    return path

def select_device(device='cpu'):
    if device.lower() == 'cpu':
        return torch.device('cpu')
    else:
        return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def plot_loss_curve(epochs, losses, save_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses, 'b-', linewidth=2, label='train loss')
    plt.title('train loss curve', fontsize=14)  # 中文标题
    plt.xlabel('(Epoch)', fontsize=12)  # 中文x轴
    plt.ylabel('loss value', fontsize=12)  # 中文y轴
    plt.grid(linestyle='--', alpha=0.7) # 增加网格线
    plt.legend(fontsize=12) # 增加图例
    plt.tight_layout()
    
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "loss_curve.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

if __name__ == '__main__':
    # 正确传入epochs列表（与losses长度一致）
    epochs = list(range(1, 11))  # [1,2,...,10]，共10个epoch
    losses = [10,9,8,7,6,5,3,2,1,0]  # 10个损失值
    save_dir = r"E:\DMH\Shanghai\dmh_categorization"  # 使用原始字符串处理路径
    plot_loss_curve(epochs, losses, save_dir)
