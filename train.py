import os
import time
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import torch
from utils.Mydataset import MyDataset
from utils.net import ResNet18 as Model
from tqdm import tqdm
import csv
from utils.confusion_maxtrix import ConfusionMatrix as confusion_matrix
from pathlib import Path
import glob
import re
import sys
import argparse
from utils.general import increment_path, select_device, plot_loss_curve
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 允许重复加载 OpenMP

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))


def train(opt):
    epochs, batch_size, device, project, name, labels_classes, patience = opt.epochs, opt.batch_size, opt.device, opt.project, opt.name, opt.labels_classes, opt.patience
    # 记录运行时间
    run_time = time.strftime("%Y_%m_%d_%H_%M")

    print("#####################Start load train and val datasets.#########################")
    train_dataset = MyDataset(train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dataset = MyDataset(train=False) # 需要增加数据增强
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True) 
    print("#####################Finish load train and val datasets.#########################")
    save_dir = str(increment_path(project/name))
    device = select_device(device)
    model = Model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    class_weights = torch.tensor([0.5, 0.5])
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights).to(device)
    count = 0
    best_val_acc = 0
    best_val_F1 = 0
    train_loss = []
    for epoch in range(1, epochs):
        print("###########"+'Epoch {}/{}'.format(epoch, epochs)+"##########")
        # 训练阶段
        model.train()
        epochs_loss = 0
        train_pbar = tqdm(train_dataloader, desc=f"Train Epoch {epoch}/{epochs}")
        for i, (images, labels) in enumerate(train_pbar):
            images, labels = images.to(device).float(), labels.to(device).float()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()
            train_pbar.set_postfix(loss=loss.item())
            epochs_loss += loss.item()
        scheduler.step()
        train_loss.append(epochs_loss/len(train_dataloader))


        # 验证阶段
        model.eval()
        val_pbar = tqdm(val_dataloader, desc=f"Val Epoch {epoch}/{epochs}")

        all_labels = []
        all_preds = []
        cm = confusion_matrix(labels_classes)
        with torch.no_grad():
            correct = 0
            total = 0
            for i, (images, labels) in enumerate(val_pbar):
                images, labels = images.to(device).float(), labels.to(device).float()
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)

                all_labels.append(int(labels.cpu().numpy()))
                all_preds.append(predicted.cpu().numpy())

                total += labels.size(0)
                correct += (predicted == labels.long()).sum().item()
                val_pbar.set_postfix(acc=100 * correct / total)
    
        cm.update(all_preds, all_labels)
        p_r_f1_acc = cm.plot_combined(path=os.path.join(save_dir, f"epoch_{epoch}_metrics.png"))
        p_r_f1_acc['epoch'] = epoch
        path = os.path.join(save_dir,"results.csv") 

        with open(path, 'a', newline='', encoding='utf-8') as f:
            # 正确初始化writer变量
            writer = csv.DictWriter(f, fieldnames=['epoch', 'val/Precision', 'val/Recall', 'val/F1-Score', 'val/acc'])
            
            # 仅在文件不存在时写入表头
            if not os.path.exists(path) or os.stat(path).st_size == 0:
                writer.writeheader()
            
            writer.writerow(p_r_f1_acc)
        torch.save(model.state_dict(), os.path.join(save_dir, f"weights/last.pt"))
        improvement_threshold = 0.001
        current_acc = p_r_f1_acc['val/acc']
        current_F1 = p_r_f1_acc['val/F1-Score']
        if (current_acc > best_val_acc + improvement_threshold) or (current_F1 > best_val_F1 + improvement_threshold):
            best_val_acc = current_acc
            best_val_F1 = current_F1
            torch.save(model.state_dict(), os.path.join(save_dir, f"weights/best.pt"))
        else:
            count = count + 1
            if count > patience:
                plot_loss_curve(range(1, epoch+1), train_loss, save_dir)
                print("No improvement in 10 epochs, early stopping...")
                break
        print(f"Epoch {epoch}/{epochs} Acc: {100 * correct / total}")
    print("#####################Finish train.#")



def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200, help='总训练轮次')
    parser.add_argument('--batch_size', type=int, default=4, help='train set proportion')
    parser.add_argument('--device', default='0', help="cuda device or cpu")
    parser.add_argument('--project', default=ROOT / 'train/runs', help='训练结果保存路径')
    parser.add_argument('--name', default='exp', help='保存路径名称 /project/exp')
    parser.add_argument('--labels_classes', default=["blue", "red"], help='标签列表')
    parser.add_argument('--patience', default=3, help='早停机制，当验证集指标连续patience个epoch不提升时，停止训练')
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_opt()
    train(opt)
    