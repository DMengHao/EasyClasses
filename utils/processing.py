import cv2
import os
import random
import pickle
import sys
from pathlib import Path
from tqdm import tqdm
import argparse


# 设置路径
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))


def is_image_file(filename):
    """判断文件是否为图片"""
    return filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'))

def count_total_images(path):
    """统计所有图片文件的总数，用于进度条初始化"""
    total = 0
    for root, dirs, _ in os.walk(path):
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            # 只统计图片文件
            for file in os.listdir(dir_path):
                if is_image_file(file):
                    total += 1
    return total

def resize_with_aspect_ratio(image, target_size, pad_value=0):
    """
    保持纵横比缩放，空白处填充
    pad_value: 填充值，默认为0（黑色），RGB图像可用(255,255,255)表示白色
    """
    h, w = image.shape[:2]
    target_w, target_h = target_size
    
    # 计算缩放比例
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # 缩放图像
    resized = cv2.resize(image, (new_w, new_h))
    
    # 计算填充量
    pad_w = (target_w - new_w) // 2
    pad_h = (target_h - new_h) // 2
    
    # 处理奇数情况
    pad_w2 = target_w - new_w - pad_w
    pad_h2 = target_h - new_h - pad_h
    
    # 填充图像
    if len(image.shape) == 3:
        # 彩色图像
        padded = cv2.copyMakeBorder(
            resized, pad_h, pad_h2, pad_w, pad_w2,
            cv2.BORDER_CONSTANT, value=pad_value
        )
    else:
        # 灰度图像
        padded = cv2.copyMakeBorder(
            resized, pad_h, pad_h2, pad_w, pad_w2,
            cv2.BORDER_CONSTANT, value=pad_value
        )
    
    return padded

def processings(path, proportions, shape=(224, 224)):
    """
    处理图像数据，分割训练集和测试集，并保存为pickle文件
    
    参数:
        path: 图像根目录
        proportions: 训练集比例
        
    返回:
        标签列表 (class_id, class_name)
    """
    all_data = [] 
    labels = []  # (class_id, class_name)
    
    # 先检查路径是否存在
    if not os.path.exists(path):
        raise FileNotFoundError(f"路径不存在: {path}")
    
    # 统计总图片数量，用于进度条
    total_images = count_total_images(path)
    if total_images == 0:
        raise ValueError(f"在路径 {path} 下未找到任何图片文件")

    # 初始化进度条
    with tqdm(total=total_images, desc="处理图像", unit="张") as pbar:
        for root, dirs, _ in os.walk(path):
            # 为每个目录分配唯一的类别ID
            for class_id, dir_name in enumerate(dirs):
                # 确保每个类别只添加一次到标签列表
                if not any(label[1] == dir_name for label in labels):
                    labels.append((class_id, dir_name))
                
                dir_path = os.path.join(root, dir_name)
                # 处理目录中的每个图片文件
                for file in os.listdir(dir_path):
                    if is_image_file(file):
                        file_path = os.path.join(dir_path, file)
                        try:
                            # 读取图片
                            img = cv2.imread(file_path)
                            img = resize_with_aspect_ratio(img, shape)
                            img = img/255.0  # 归一化到0-1之间
                            if img is not None:
                                all_data.append((class_id, img))
                            else:
                                print(f"警告: 无法读取图片 {file_path}")
                        except Exception as e:
                            print(f"处理图片 {file_path} 时出错: {str(e)}")
                        finally:
                            pbar.update(1)  # 更新进度条
    
    # 打乱数据顺序
    random.shuffle(all_data)
    
    # 分割训练集和测试集
    split_idx = int(len(all_data) * proportions)
    train_data = all_data[:split_idx]
    test_data = all_data[split_idx:]
    
    # 创建数据保存目录
    data_dir = os.path.join(ROOT, "data")
    os.makedirs(data_dir, exist_ok=True)  # 使用exist_ok避免目录已存在的错误
    
    # 保存训练集
    train_path = os.path.join(data_dir, "train_data_cache.pkl")
    with open(train_path, "wb") as f:
        pickle.dump(train_data, f)
    
    # 保存测试集
    test_path = os.path.join(data_dir, "val_data_cache.pkl")
    with open(test_path, "wb") as f:
        pickle.dump(test_data, f)
    
    print(f"\n数据处理完成: 共 {len(all_data)} 张图片")
    print(f"训练集: {len(train_data)} 张, 测试集: {len(test_data)} 张")
    print(f"数据已保存至: {data_dir}")
    
    return labels


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=r"C:\Users\hhkj\Desktop\images", help='path to dataset')
    parser.add_argument('--proportions', type=float, default=0.8, help='train set proportion')
    parser.add_argument('--shape', type=int, nargs='+', default=[224, 224], help='image shape')
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


def main(opt):
    try:
        print("开始图像预处理...")
        labels = processings(opt.data_path, opt.proportions)
        print("\n预处理完成!")
        print(f"类别标签 ({len(labels)} 个类别):")
        for class_id, class_name in labels:
            print(f"  类别 {class_id}: {class_name}")
    except Exception as e:
        print(f"处理过程出错: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)