#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
深度学习基础示例代码
==================
本代码演示深度学习的基本组件和工作流程:
1. 搭建简单神经网络
2. 使用PyTorch训练图像分类模型
3. 可视化训练过程和结果
4. 模型保存与加载
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import time
import os

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 检查是否有GPU可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")


# 定义一个简单的卷积神经网络
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 第一个卷积层, 1个输入通道, 32个输出通道, 3x3卷积核
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        # 第二个卷积层, 32个输入通道, 64个输出通道, 3x3卷积核
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # 池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 全连接层1, 64*7*7个输入特征, 128个输出特征
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        # 全连接层2, 128个输入特征, 10个输出特征(对应10个数字类别)
        self.fc2 = nn.Linear(128, 10)
        # Dropout层, 用于防止过拟合
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # 第一个卷积层 + 激活函数 + 池化
        x = self.pool(F.relu(self.conv1(x)))
        # 第二个卷积层 + 激活函数 + 池化
        x = self.pool(F.relu(self.conv2(x)))
        # 将特征图展平为向量
        x = x.view(-1, 64 * 7 * 7)
        # 全连接层1 + 激活函数 + Dropout
        x = self.dropout(F.relu(self.fc1(x)))
        # 全连接层2(输出层)
        x = self.fc2(x)
        return x


# 数据预处理和加载
def load_data():
    """加载MNIST数据集并进行预处理"""
    print("="*50)
    print("数据加载与预处理")
    print("="*50)
    
    # 数据变换: 转换为张量, 标准化
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST数据集的均值和标准差
    ])
    
    # 加载训练集
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # 加载测试集
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    print(f"类别数量: {len(train_dataset.classes)}")
    print(f"类别: {train_dataset.classes}")
    
    # 可视化一些样本
    plt.figure(figsize=(12, 6))
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.imshow(train_dataset.data[i], cmap='gray')
        plt.title(f"标签: {train_dataset.targets[i]}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('mnist_samples.png')
    print("MNIST样本图已保存为 mnist_samples.png")
    
    return train_loader, test_loader


# 网络结构可视化
def visualize_network():
    """可视化网络结构"""
    print("\n" + "="*50)
    print("网络结构可视化")
    print("="*50)
    
    model = SimpleCNN()
    print(model)
    
    # 统计模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"总参数量: {total_params}")
    print(f"可训练参数量: {trainable_params}")
    
    # 打印每层的参数量
    print("\n各层参数量:")
    for name, param in model.named_parameters():
        print(f"{name}: {param.numel()}")


# 训练函数
def train(model, train_loader, optimizer, criterion, epoch):
    """训练模型一个epoch"""
    model.train()  # 设置为训练模式
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # 梯度清零
        optimizer.zero_grad()
        
        # 前向传播
        output = model(data)
        
        # 计算损失
        loss = criterion(output, target)
        
        # 反向传播
        loss.backward()
        
        # 更新参数
        optimizer.step()
        
        # 累计损失
        train_loss += loss.item()
        
        # 计算准确率
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        # 打印进度
        if (batch_idx + 1) % 100 == 0:
            print(f'Epoch: {epoch} | Batch: {batch_idx+1}/{len(train_loader)} | '
                  f'Loss: {train_loss/(batch_idx+1):.4f} | '
                  f'Acc: {100.*correct/total:.2f}%')
    
    return train_loss/len(train_loader), correct/total


# 测试函数
def test(model, test_loader, criterion):
    """在测试集上评估模型"""
    model.eval()  # 设置为评估模式
    test_loss = 0
    correct = 0
    total = 0
    
    # 不计算梯度
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            # 前向传播
            output = model(data)
            
            # 计算损失
            loss = criterion(output, target)
            test_loss += loss.item()
            
            # 计算准确率
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    # 返回平均损失和准确率
    return test_loss/len(test_loader), correct/total


# 可视化预测结果
def visualize_predictions(model, test_loader):
    """可视化模型在一些测试样本上的预测结果"""
    print("\n" + "="*50)
    print("预测结果可视化")
    print("="*50)
    
    model.eval()
    
    # 获取一批测试数据
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    
    # 选择前15个样本
    images = images[:15].to(device)
    labels = labels[:15]
    
    # 预测
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
    
    # 转回CPU进行可视化
    images = images.cpu()
    predicted = predicted.cpu()
    
    # 可视化
    plt.figure(figsize=(20, 10))
    for i in range(15):
        plt.subplot(3, 5, i+1)
        plt.imshow(images[i][0], cmap='gray')
        color = 'green' if predicted[i] == labels[i] else 'red'
        plt.title(f'预测: {predicted[i]}, 实际: {labels[i]}', color=color)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('prediction_visualization.png')
    print("预测结果可视化已保存为 prediction_visualization.png")


# 可视化卷积层特征图
def visualize_feature_maps(model, test_loader):
    """可视化第一个卷积层的特征图"""
    print("\n" + "="*50)
    print("特征图可视化")
    print("="*50)
    
    # 获取一个样本
    dataiter = iter(test_loader)
    images, _ = next(dataiter)
    img = images[0:1].to(device)  # 只取第一个样本
    
    # 创建一个新模型，只包含第一个卷积层
    conv1 = model.conv1
    
    # 获取第一个卷积层的输出
    with torch.no_grad():
        features = conv1(img)
    
    # 转回CPU进行可视化
    features = features.cpu().numpy()
    
    # 可视化前16个特征图
    plt.figure(figsize=(20, 10))
    for i in range(min(16, features.shape[1])):
        plt.subplot(4, 4, i+1)
        plt.imshow(features[0, i], cmap='viridis')
        plt.title(f'特征图 #{i+1}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('feature_maps.png')
    print("特征图可视化已保存为 feature_maps.png")


# 可视化训练过程
def plot_training_history(train_losses, train_accs, test_losses, test_accs):
    """绘制训练损失和准确率的变化曲线"""
    plt.figure(figsize=(12, 5))
    
    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='训练损失')
    plt.plot(test_losses, label='测试损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.title('训练与测试损失')
    plt.legend()
    
    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='训练准确率')
    plt.plot(test_accs, label='测试准确率')
    plt.xlabel('Epoch')
    plt.ylabel('准确率')
    plt.title('训练与测试准确率')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    print("\n训练历史已保存为 training_history.png")


# 混淆矩阵可视化
def plot_confusion_matrix(model, test_loader):
    """绘制混淆矩阵"""
    print("\n" + "="*50)
    print("混淆矩阵可视化")
    print("="*50)
    
    model.eval()
    
    # 收集所有预测和真实标签
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, preds = torch.max(output, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
    
    # 计算混淆矩阵
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('混淆矩阵')
    plt.colorbar()
    
    classes = [str(i) for i in range(10)]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    
    # 添加文本标注
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.savefig('confusion_matrix.png')
    print("混淆矩阵已保存为 confusion_matrix.png")


# 训练模型
def train_model(model, train_loader, test_loader, epochs=5):
    """训练模型主函数"""
    print("\n" + "="*50)
    print("模型训练")
    print("="*50)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 记录训练过程
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []
    
    # 训练循环
    start_time = time.time()
    for epoch in range(1, epochs+1):
        # 训练一个epoch
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, epoch)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # 在测试集上评估
        test_loss, test_acc = test(model, test_loader, criterion)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        print(f'\nEpoch {epoch} | '
              f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | '
              f'Test Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.2f}%\n')
    
    # 训练时间
    total_time = time.time() - start_time
    print(f'训练完成! 总耗时: {total_time:.2f} 秒')
    
    # 可视化训练过程
    plot_training_history(train_losses, train_accs, test_losses, test_accs)
    
    return model


# 保存和加载模型
def save_and_load_model(model):
    """保存和加载模型"""
    print("\n" + "="*50)
    print("模型保存与加载")
    print("="*50)
    
    # 创建模型目录
    model_dir = 'saved_models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # 保存模型
    model_path = os.path.join(model_dir, 'mnist_cnn.pth')
    torch.save(model.state_dict(), model_path)
    print(f"模型已保存到 {model_path}")
    
    # 加载模型到新的实例
    loaded_model = SimpleCNN().to(device)
    loaded_model.load_state_dict(torch.load(model_path))
    loaded_model.eval()
    print("模型已成功加载")
    
    return loaded_model


def main():
    """主函数"""
    print("\n" + "*"*70)
    print("*" + " "*28 + "深度学习基础" + " "*28 + "*")
    print("*"*70)
    
    # 加载数据
    train_loader, test_loader = load_data()
    
    # 可视化网络结构
    visualize_network()
    
    # 创建模型实例
    model = SimpleCNN().to(device)
    
    # 训练模型
    model = train_model(model, train_loader, test_loader, epochs=5)
    
    # 可视化预测结果
    visualize_predictions(model, test_loader)
    
    # 可视化混淆矩阵
    plot_confusion_matrix(model, test_loader)
    
    # 可视化特征图
    visualize_feature_maps(model, test_loader)
    
    # 保存和加载模型
    loaded_model = save_and_load_model(model)
    
    print("\n" + "*"*70)
    print("*" + " "*24 + "深度学习基础演示完成" + " "*24 + "*")
    print("*"*70)


if __name__ == "__main__":
    main() 