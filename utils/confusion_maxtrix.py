import os
import numpy as np
from matplotlib import pyplot as plt
import csv

class ConfusionMatrix(object):
    def __init__(self, labels=None):
        self.labels = labels
        self.num_classes = len(labels)
        self.matrix = np.zeros((self.num_classes, self.num_classes))  # 修正变量名拼写错误（maxtrix→matrix）

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1

    def plot_combined(self, path=None):
        """将混淆矩阵和评价指标绘制到同一张图片"""
        matrix = self.matrix.copy()
        sum_TP = np.trace(matrix)  # 计算对角线元素和（正确预测总数）
        total = np.sum(matrix)
        acc = round(sum_TP / total, 4) if total != 0 else 0

        # 创建1行2列的子图布局
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle(f"model evaluation (all classes accuature: {acc}%)", fontsize=16)

        # ------------------- 左侧：混淆矩阵 -------------------
        im = ax1.imshow(matrix, cmap=plt.cm.Blues)
        ax1.set_xticks(range(self.num_classes))
        ax1.set_xticklabels(self.labels, rotation=45, ha='right')
        ax1.set_yticks(range(self.num_classes))
        ax1.set_yticklabels(self.labels)
        ax1.set_xlabel("True")
        ax1.set_ylabel("Predicted")
        ax1.set_title("Confusion Matrix")

        # 添加数值标注
        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                info = int(matrix[y, x])
                ax1.text(x, y, info,
                         horizontalalignment='center',
                         verticalalignment='center',
                         color="white" if matrix[y, x] > thresh else "black")

        # 添加颜色条
        fig.colorbar(im, ax=ax1, shrink=0.8)

        # ------------------- 右侧：评价指标表格 -------------------
        columns = ['classes', 'Precision', 'Recall', 'F1-Score']
        cell_text = []
        for i in range(self.num_classes):
            TP = matrix[i, i]
            FP = sum(matrix[i, :]) - TP
            FN = sum(matrix[:, i]) - TP
            
            # 计算指标（避免除零错误）
            Precision = round(TP / (TP + FP), 4) if (TP + FP) != 0 else 0
            Recall = round(TP / (TP + FN), 4) if (TP + FN) != 0 else 0
            F1 = round(2 * Precision * Recall / (Precision + Recall), 4) if (Precision + Recall) != 0 else 0
            
            cell_text.append([self.labels[i], Precision, Recall, F1])

        # 绘制表格
        table = ax2.table(cellText=cell_text, colLabels=columns, loc='center')
        ax2.axis('off')  # 隐藏坐标轴
        ax2.set_title("Classification indicator details")
        
        # 美化表格
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)  # 调整表格大小

        # 调整布局
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # 预留标题空间
        plt.savefig(path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"合并图表已保存至: {path}")
        return {
            'val/Precision': Precision,
            'val/Recall': Recall,
            'val/F1-Score': F1,
            'val/acc': acc
        }


if __name__ == '__main__':
    # 确保保存目录存在
    save_dir = r"E:\DMH\Shanghai\dmh_categorization\results"
    os.makedirs(save_dir, exist_ok=True)
    
    # 初始化混淆矩阵
    cm = ConfusionMatrix(labels=["red", "blue"])
    
    # 更新数据（示例预测和真实标签）
    cm.update(preds=[0, 0, 1, 0], labels=[0, 0, 1, 1])
    
    # 绘制并保存合并图表
    p_r_f1_acc = cm.plot_combined(path=os.path.join(save_dir, "combined_metrics.png"))
    path = r'E:\DMH\Shanghai\dmh_categorization\results\epoch_metrics.csv'
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'Precision', 'Recall', 'F1-Score', 'acc'])
        writer.writeheader()
        writer.writerow(p_r_f1_acc)
    print(f"合并图表已保存至: {path}")
    