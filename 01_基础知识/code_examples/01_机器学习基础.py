#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
机器学习基础示例代码
====================
本代码演示机器学习的基本工作流程：
1. 数据加载与预处理
2. 特征工程
3. 模型训练与评估
4. 模型保存与加载
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import joblib
import os

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def load_data():
    """加载数据集并进行简单探索"""
    print("="*50)
    print("数据加载与探索")
    print("="*50)
    
    # 加载鸢尾花数据集
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    
    # 转换为DataFrame以便查看
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    df['species'] = [target_names[i] for i in y]
    
    # 数据集基本信息
    print(f"数据集形状: {X.shape}")
    print(f"特征名称: {feature_names}")
    print(f"目标类别: {target_names}")
    print("\n数据集前5行:")
    print(df.head())
    
    # 数据集统计信息
    print("\n数据集统计信息:")
    print(df.describe())
    
    # 各类别样本数量统计
    print("\n各类别样本数量:")
    print(df['species'].value_counts())
    
    # 可视化数据
    plt.figure(figsize=(12, 5))
    
    # 散点图：花瓣长度 vs 花瓣宽度
    plt.subplot(1, 2, 1)
    for target, target_name in enumerate(target_names):
        target_data = df[df['target'] == target]
        plt.scatter(target_data['petal length (cm)'], 
                   target_data['petal width (cm)'], 
                   label=target_name)
    plt.xlabel('花瓣长度 (cm)')
    plt.ylabel('花瓣宽度 (cm)')
    plt.title('花瓣长度 vs 花瓣宽度')
    plt.legend()
    
    # 散点图：萼片长度 vs 萼片宽度
    plt.subplot(1, 2, 2)
    for target, target_name in enumerate(target_names):
        target_data = df[df['target'] == target]
        plt.scatter(target_data['sepal length (cm)'], 
                   target_data['sepal width (cm)'], 
                   label=target_name)
    plt.xlabel('萼片长度 (cm)')
    plt.ylabel('萼片宽度 (cm)')
    plt.title('萼片长度 vs 萼片宽度')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('iris_visualization.png')
    print("数据可视化已保存到 iris_visualization.png")
    
    return X, y, feature_names, target_names


def preprocess_data(X, y):
    """数据预处理与分割"""
    print("\n" + "="*50)
    print("数据预处理")
    print("="*50)
    
    # 分割数据集为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"训练集形状: {X_train.shape}")
    print(f"测试集形状: {X_test.shape}")
    
    # 标准化特征
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("数据标准化前后对比 (训练集前5个样本):")
    print("标准化前:", X_train[:5, 0])
    print("标准化后:", X_train_scaled[:5, 0])
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_models(X_train, X_test, y_train, y_test):
    """训练多个模型并比较性能"""
    print("\n" + "="*50)
    print("模型训练与评估")
    print("="*50)
    
    # 定义要比较的模型
    models = {
        "逻辑回归": LogisticRegression(max_iter=1000, random_state=42),
        "随机森林": RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    # 训练并评估每个模型
    for name, model in models.items():
        print(f"\n训练模型: {name}")
        model.fit(X_train, y_train)
        
        # 预测
        y_pred = model.predict(X_test)
        
        # 计算准确率
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
        
        print(f"{name} 准确率: {accuracy:.4f}")
        
        # 打印分类报告
        print("\n分类报告:")
        print(classification_report(y_test, y_pred))
        
        # 绘制混淆矩阵
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'{name} - 混淆矩阵')
        plt.colorbar()
        tick_marks = np.arange(len(iris.target_names))
        plt.xticks(tick_marks, iris.target_names, rotation=45)
        plt.yticks(tick_marks, iris.target_names)
        
        # 在混淆矩阵中显示数字
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        plt.savefig(f'{name}_confusion_matrix.png')
        print(f"{name} 混淆矩阵已保存为 {name}_confusion_matrix.png")
    
    # 比较模型性能
    plt.figure(figsize=(10, 6))
    plt.bar(results.keys(), results.values())
    plt.xlabel('模型')
    plt.ylabel('准确率')
    plt.title('模型性能比较')
    plt.ylim(0.8, 1.0)  # 设置y轴范围以更好地显示差异
    for i, (model, accuracy) in enumerate(results.items()):
        plt.text(i, accuracy + 0.01, f'{accuracy:.4f}', ha='center')
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    print("\n模型比较图已保存为 model_comparison.png")
    
    # 返回性能最好的模型
    best_model_name = max(results, key=results.get)
    best_model = models[best_model_name]
    
    print(f"\n性能最好的模型是: {best_model_name}，准确率: {results[best_model_name]:.4f}")
    
    return best_model


def save_and_load_model(model, scaler, X_test, y_test):
    """保存和加载模型"""
    print("\n" + "="*50)
    print("模型保存与加载")
    print("="*50)
    
    # 创建模型目录
    model_dir = 'saved_models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # 创建模型管道
    pipeline = Pipeline([
        ('scaler', scaler),
        ('model', model)
    ])
    
    # 保存模型
    model_path = os.path.join(model_dir, 'iris_model.pkl')
    joblib.dump(pipeline, model_path)
    print(f"模型已保存到 {model_path}")
    
    # 加载模型
    loaded_pipeline = joblib.load(model_path)
    print("模型已成功加载")
    
    # 用加载的模型进行预测
    y_pred = loaded_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"加载的模型准确率: {accuracy:.4f}")
    
    return loaded_pipeline


def prediction_example(model, feature_names, target_names):
    """演示模型预测新样本"""
    print("\n" + "="*50)
    print("模型预测示例")
    print("="*50)
    
    # 创建一些新样本数据
    new_samples = [
        [5.1, 3.5, 1.4, 0.2],  # 可能是Setosa
        [6.3, 3.3, 4.7, 1.6],  # 可能是Versicolor
        [6.5, 3.0, 5.2, 2.0]   # 可能是Virginica
    ]
    
    print("新样本特征:")
    for i, sample in enumerate(new_samples):
        print(f"样本 {i+1}: {dict(zip(feature_names, sample))}")
    
    # 预测
    predictions = model.predict(new_samples)
    
    print("\n预测结果:")
    for i, pred in enumerate(predictions):
        print(f"样本 {i+1} 预测为: {target_names[pred]}")
    
    # 获取预测概率
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(new_samples)
        print("\n预测概率:")
        for i, proba in enumerate(probabilities):
            print(f"样本 {i+1} 预测概率:")
            for j, p in enumerate(proba):
                print(f"  - {target_names[j]}: {p:.4f}")


def feature_importance(model, feature_names):
    """分析特征重要性"""
    print("\n" + "="*50)
    print("特征重要性分析")
    print("="*50)
    
    # 检查模型是否有特征重要性属性
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        
        # 排序特征重要性
        indices = np.argsort(importances)[::-1]
        
        print("特征重要性排序:")
        for i, idx in enumerate(indices):
            print(f"{i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
        
        # 可视化特征重要性
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(importances)), 
                [importances[i] for i in indices],
                align='center')
        plt.xticks(range(len(importances)), 
                  [feature_names[i] for i in indices], 
                  rotation=90)
        plt.title('特征重要性')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        print("特征重要性图已保存为 feature_importance.png")
    else:
        print("当前模型不支持特征重要性分析")


def main():
    """主函数"""
    print("\n" + "*"*70)
    print("*" + " "*28 + "机器学习工作流程" + " "*28 + "*")
    print("*"*70)
    
    # 加载并探索数据
    iris = load_iris()
    X, y, feature_names, target_names = load_data()
    
    # 数据预处理
    X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)
    
    # 训练模型
    best_model = train_models(X_train, X_test, y_train, y_test)
    
    # 保存和加载模型
    loaded_model = save_and_load_model(best_model, scaler, X_test, y_test)
    
    # 模型预测示例
    prediction_example(best_model, feature_names, target_names)
    
    # 特征重要性分析
    feature_importance(best_model, feature_names)
    
    print("\n" + "*"*70)
    print("*" + " "*24 + "机器学习工作流程演示完成" + " "*24 + "*")
    print("*"*70)


if __name__ == "__main__":
    main() 