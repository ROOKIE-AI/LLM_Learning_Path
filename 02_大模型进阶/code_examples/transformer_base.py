#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Transformer基础实现
================
本代码实现了Transformer模型的核心组件:
1. 多头自注意力机制 (Multi-Head Self-Attention)
2. 位置编码 (Positional Encoding)
3. 前馈神经网络 (Feed-Forward Network)
4. Transformer编码器和解码器
5. 简单的翻译任务示例
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import copy
import time
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import re
from collections import Counter
import matplotlib as mpl


# 设置随机种子，保证结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 检查是否有GPU可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class PositionalEncoding(nn.Module):
    """
    位置编码实现
    """
    def __init__(self, d_model, max_seq_length=5000):
        """
        初始化位置编码
        参数:
            d_model: 模型维度
            max_seq_length: 最大序列长度
        """
        super(PositionalEncoding, self).__init__()
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_seq_length, d_model) # 创建一个形状为(max_seq_length, d_model)的零矩阵
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1) # 创建一个形状为(max_seq_length, 1)的矩阵，表示位置
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # 计算位置编码的缩放因子
        
        # 计算位置编码
        pe[:, 0::2] = torch.sin(position * div_term) # 计算偶数位置的正弦值
        pe[:, 1::2] = torch.cos(position * div_term) # 计算奇数位置的余弦值
        
        # 注册为buffer（不是模型参数）
        self.register_buffer('pe', pe.unsqueeze(0)) # 将位置编码矩阵pe注册为buffer，不会参与梯度更新
    
    def forward(self, x):
        """
        参数:
            x: 输入张量，形状 [batch_size, seq_length, d_model]
        """
        return x + self.pe[:, :x.size(1), :]


# 多头注意力机制实现
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        """
        初始化多头注意力机制
        参数:
            d_model: 模型维度
            num_heads: 头数
            dropout: 丢弃率
        """
        super(MultiHeadAttention, self).__init__()
        # 确保模型维度可以被头数整除
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度
        
        # 线性投影层: 将输入的d_model维度的张量投影为d_model维度的张量
        self.W_q = nn.Linear(d_model, d_model)  
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        计算缩放点积注意力
        参数:
            Q: 查询矩阵，形状 [batch_size, num_heads, seq_length, d_k]
            K: 键矩阵，形状 [batch_size, num_heads, seq_length, d_k]
            V: 值矩阵，形状 [batch_size, num_heads, seq_length, d_k]
            mask: 掩码，形状 [batch_size, 1, 1, seq_length] 或 [batch_size, 1, seq_length, seq_length]
        """
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 应用掩码（如果提供）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 应用softmax获取注意力权重
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 计算输出
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def split_heads(self, x):
        """
        将张量分割为多个头
        参数:
            x: 输入张量，形状 [batch_size, seq_length, d_model]
        """
        batch_size, seq_length = x.size(0), x.size(1)
        
        # 重塑张量为 [batch_size, seq_length, num_heads, d_k]
        x = x.view(batch_size, seq_length, self.num_heads, self.d_k)
        
        # 转置为 [batch_size, num_heads, seq_length, d_k]
        return x.transpose(1, 2)
    
    def combine_heads(self, x):
        """
        将多个头合并回来
        参数:
            x: 输入张量，形状 [batch_size, num_heads, seq_length, d_k]
        """
        batch_size, _, seq_length, _ = x.size()
        
        # 转置为 [batch_size, seq_length, num_heads, d_k]
        x = x.transpose(1, 2)
        
        # 重塑为 [batch_size, seq_length, d_model]
        return x.contiguous().view(batch_size, seq_length, self.d_model)
    
    def forward(self, query, key, value, mask=None):
        """
        前向传播
        参数:
            query: 查询张量，形状 [batch_size, seq_length, d_model]
            key: 键张量，形状 [batch_size, seq_length, d_model]
            value: 值张量，形状 [batch_size, seq_length, d_model]
            mask: 掩码，形状 [batch_size, 1, 1, seq_length] 或 [batch_size, 1, seq_length, seq_length]
        """
        batch_size = query.size(0)
        
        # 线性投影
        Q = self.W_q(query) # 将输入的query张量投影为d_model维度的张量
        K = self.W_k(key) # 将输入的key张量投影为d_model维度的张量
        V = self.W_v(value) # 将输入的value张量投影为d_model维度的张量
        
        # 分割多头
        Q = self.split_heads(Q) # 将输入的Q张量分割为多个头
        K = self.split_heads(K) # 将输入的K张量分割为多个头
        V = self.split_heads(V) # 将输入的V张量分割为多个头
        
        # 缩放点积注意力
        output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask) # 计算缩放点积注意力
        
        # 合并多头
        output = self.combine_heads(output) # 将多个头合并回来
        
        # 最终线性投影
        output = self.W_o(output) # 将输入的output张量投影为d_model维度的张量
        
        return output, attention_weights


# 前馈神经网络实现
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        初始化前馈神经网络
        参数:
            d_model: 模型维度
            d_ff: 前馈神经网络的中间维度
            dropout: 丢弃率
        """
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff) # 创建一个形状为(d_model, d_ff)的线性层
        self.linear2 = nn.Linear(d_ff, d_model) # 创建一个形状为(d_ff, d_model)的线性层
        self.dropout = nn.Dropout(dropout) # 创建一个形状为(dropout)的dropout层
    
    def forward(self, x):
        """
        前向传播
        参数:
            x: 输入张量，形状 [batch_size, seq_length, d_model]
        """
        return self.linear2(self.dropout(F.relu(self.linear1(x)))) # 将输入的x张量通过线性层和dropout层，然后通过relu激活函数，最后通过线性层投影为d_model维度的张量    


# 编码器层实现
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        初始化编码器层
        参数:
            d_model: 模型维度
            num_heads: 头数
            d_ff: 前馈神经网络的中间维度
            dropout: 丢弃率
        """
        super(EncoderLayer, self).__init__()
        
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout) # 创建一个多头注意力机制
        self.feed_forward = FeedForward(d_model, d_ff, dropout) # 创建一个前馈神经网络
        
        self.norm1 = nn.LayerNorm(d_model) # 创建一个形状为(d_model)的层归一化层
        self.norm2 = nn.LayerNorm(d_model) # 创建一个形状为(d_model)的层归一化层
        
        self.dropout = nn.Dropout(dropout) # 创建一个形状为(dropout)的dropout层 
    
    def forward(self, x, mask=None):
        """
        前向传播
        参数:
            x: 输入张量，形状 [batch_size, seq_length, d_model]
            mask: 掩码，形状 [batch_size, 1, 1, seq_length]
        """
        # 自注意力子层
        attn_output, _ = self.self_attn(x, x, x, mask) # 计算自注意力
        x = self.norm1(x + self.dropout(attn_output)) # 将输入的x张量加上自注意力输出，然后通过层归一化
        
        # 前馈网络子层
        ff_output = self.feed_forward(x) # 计算前馈神经网络
        x = self.norm2(x + self.dropout(ff_output)) # 将输入的x张量加上前馈神经网络输出，然后通过层归一化
        
        return x


# 解码器层实现
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        初始化解码器层
        参数:
            d_model: 模型维度
            num_heads: 头数
            d_ff: 前馈神经网络的中间维度
            dropout: 丢弃率
        """
        super(DecoderLayer, self).__init__()
        
        # 自注意力
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout) # 创建一个多头注意力机制
        # 编码器-解码器注意力
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout) # 创建一个多头注意力机制
        # 前馈网络
        self.feed_forward = FeedForward(d_model, d_ff, dropout) # 创建一个前馈神经网络
        
        self.norm1 = nn.LayerNorm(d_model) # 创建一个形状为(d_model)的层归一化层
        self.norm2 = nn.LayerNorm(d_model) # 创建一个形状为(d_model)的层归一化层
        self.norm3 = nn.LayerNorm(d_model) # 创建一个形状为(d_model)的层归一化层
        
        self.dropout = nn.Dropout(dropout) # 创建一个形状为(dropout)的dropout层
    
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        """
        前向传播
        参数:
            x: 输入张量，形状 [batch_size, seq_length, d_model]
            enc_output: 编码器输出，形状 [batch_size, src_seq_length, d_model]
            src_mask: 源序列掩码，形状 [batch_size, 1, 1, src_seq_length]
            tgt_mask: 目标序列掩码，形状 [batch_size, 1, tgt_seq_length, tgt_seq_length]
        """
        # 自注意力子层
        attn1_output, _ = self.self_attn(x, x, x, tgt_mask) # 计算自注意力
        x = self.norm1(x + self.dropout(attn1_output)) # 将输入的x张量加上自注意力输出，然后通过层归一化
        
        # 编码器-解码器注意力子层
        attn2_output, _ = self.cross_attn(x, enc_output, enc_output, src_mask) # 计算编码器-解码器注意力
        x = self.norm2(x + self.dropout(attn2_output)) # 将输入的x张量加上编码器-解码器注意力输出，然后通过层归一化
        
        # 前馈网络子层
        ff_output = self.feed_forward(x) # 计算前馈神经网络
        x = self.norm3(x + self.dropout(ff_output)) # 将输入的x张量加上前馈神经网络输出，然后通过层归一化
        
        return x


# 完整编码器实现
class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        """
        初始化编码器
        参数:
            num_layers: 层数
            d_model: 模型维度
            num_heads: 头数
            d_ff: 前馈神经网络的中间维度
            dropout: 丢弃率
        """
        super(Encoder, self).__init__()
        
        self.positional_encoding = PositionalEncoding(d_model) # 创建一个位置编码
        self.dropout = nn.Dropout(dropout) # 创建一个形状为(dropout)的dropout层
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]) # 创建一个包含num_layers个编码器层的列表
        self.norm = nn.LayerNorm(d_model) # 创建一个形状为(d_model)的层归一化层

    
    def forward(self, x, mask=None):
        """
        前向传播
        参数:
            x: 输入张量，形状 [batch_size, seq_length, d_model]
            mask: 掩码，形状 [batch_size, 1, 1, seq_length]
        """
        x = self.positional_encoding(x) # 将输入的x张量加上位置编码
        x = self.dropout(x) # 将输入的x张量通过dropout层
        
        for layer in self.layers:
            x = layer(x, mask) # 将输入的x张量通过编码器层
        
        return self.norm(x) # 将输入的x张量通过层归一化


# 完整解码器实现
class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        """
        初始化解码器
        参数:
            num_layers: 层数
            d_model: 模型维度
            num_heads: 头数
            d_ff: 前馈神经网络的中间维度
            dropout: 丢弃率
        """
        super(Decoder, self).__init__()
        
        self.positional_encoding = PositionalEncoding(d_model) # 创建一个位置编码
        self.dropout = nn.Dropout(dropout) # 创建一个形状为(dropout)的dropout层
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]) # 创建一个包含num_layers个解码器层的列表
        self.norm = nn.LayerNorm(d_model) # 创建一个形状为(d_model)的层归一化层
    
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        """
        前向传播
        参数:
            x: 输入张量，形状 [batch_size, seq_length, d_model]
            enc_output: 编码器输出，形状 [batch_size, src_seq_length, d_model]
            src_mask: 源序列掩码，形状 [batch_size, 1, 1, src_seq_length]
            tgt_mask: 目标序列掩码，形状 [batch_size, 1, tgt_seq_length, tgt_seq_length]
        """
        x = self.positional_encoding(x) # 将输入的x张量加上位置编码
        x = self.dropout(x) # 将输入的x张量通过dropout层
        
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask) # 将输入的x张量通过解码器层
        
        return self.norm(x) # 将输入的x张量通过层归一化


# 完整Transformer模型
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, num_layers=6, d_ff=2048, dropout=0.1):
        """
        初始化Transformer模型
        参数:
            src_vocab_size: 源词汇表大小
            tgt_vocab_size: 目标词汇表大小
            d_model: 模型维度
            num_heads: 头数
            num_layers: 层数
            d_ff: 前馈神经网络的中间维度
            dropout: 丢弃率
        """
        super(Transformer, self).__init__()
        
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model) # 创建一个形状为(src_vocab_size, d_model)的嵌入层
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model) # 创建一个形状为(tgt_vocab_size, d_model)的嵌入层
        
        self.encoder = Encoder(num_layers, d_model, num_heads, d_ff, dropout) # 创建一个编码器
        self.decoder = Decoder(num_layers, d_model, num_heads, d_ff, dropout) # 创建一个解码器
        
        self.final_layer = nn.Linear(d_model, tgt_vocab_size) # 创建一个形状为(d_model, tgt_vocab_size)的线性层
    
    def encode(self, src, src_mask=None):
        """编码源序列"""
        src_embedded = self.encoder_embedding(src) * math.sqrt(self.encoder_embedding.embedding_dim) # 将输入的src张量通过嵌入层，然后乘以math.sqrt(self.encoder_embedding.embedding_dim)
        return self.encoder(src_embedded, src_mask) # 将输入的src_embedded张量通过编码器
    
    def decode(self, tgt, memory, src_mask=None, tgt_mask=None):
        """解码目标序列"""
        tgt_embedded = self.decoder_embedding(tgt) * math.sqrt(self.decoder_embedding.embedding_dim) # 将输入的tgt张量通过嵌入层，然后乘以math.sqrt(self.decoder_embedding.embedding_dim)
        return self.decoder(tgt_embedded, memory, src_mask, tgt_mask) # 将输入的tgt_embedded张量通过解码器
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        前向传播
        参数:
            src: 源序列，形状 [batch_size, src_seq_length]
            tgt: 目标序列，形状 [batch_size, tgt_seq_length]
            src_mask: 源序列掩码，形状 [batch_size, 1, 1, src_seq_length]
            tgt_mask: 目标序列掩码，形状 [batch_size, 1, tgt_seq_length, tgt_seq_length]
        """
        # 编码器前向传播
        enc_output = self.encode(src, src_mask) # 将输入的src张量通过编码器
        
        # 解码器前向传播
        dec_output = self.decode(tgt, enc_output, src_mask, tgt_mask) # 将输入的tgt张量通过解码器
        
        # 线性层和softmax
        output = self.final_layer(dec_output) # 将输入的dec_output张量通过线性层和softmax
        
        return output


# 创建掩码
def create_masks(src, tgt=None):
    """
    创建源序列和目标序列的掩码
    参数:
        src: 源序列，形状 [batch_size, src_seq_length]
        tgt: 目标序列，形状 [batch_size, tgt_seq_length]
    """
    # 源序列填充掩码 - 隐藏填充符号
    src_mask = (src != 0).unsqueeze(1).unsqueeze(2) # 将输入的src张量中填充符号隐藏
    
    if tgt is not None:
        # 目标序列填充掩码
        tgt_padding_mask = (tgt != 0).unsqueeze(1).unsqueeze(2) # 将输入的tgt张量中填充符号隐藏
        
        # 目标序列后续词掩码 - 防止模型看到未来信息
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool().to(device) # 创建一个形状为(1, seq_length, seq_length)的掩码，防止模型看到未来信息
        
        # 合并两个掩码
        tgt_mask = tgt_padding_mask & nopeak_mask # 将输入的tgt_padding_mask张量和nopeak_mask张量进行按位与运算
        
        return src_mask, tgt_mask
    
    return src_mask


# 可视化位置编码
def visualize_positional_encoding():
    """Visualize positional encoding"""
    print("\n" + "="*50)
    print("Positional Encoding Visualization")
    print("="*50)
    
    # 创建位置编码对象
    d_model = 512 # 模型维度
    max_seq_length = 100 # 最大序列长度
    pe = PositionalEncoding(d_model) # 创建一个位置编码对象
    
    # 获取位置编码矩阵
    pos_encoding = pe.pe.squeeze(0).numpy() # 获取位置编码矩阵
    
    # 可视化部分位置编码
    plt.figure(figsize=(15, 5)) # 创建一个大小为(15, 5)的图像
    plt.imshow(pos_encoding[:max_seq_length, :20], aspect='auto', cmap='viridis') # 可视化部分位置编码
    plt.xlabel('Encoding Dimension') # 设置x轴标签
    plt.ylabel('Position') # 设置y轴标签
    plt.title('Transformer Positional Encoding (First 20 dimensions)') # 设置标题
    plt.colorbar() # 显示颜色条
    plt.show() # 显示图像


# 可视化自注意力
def visualize_self_attention():
    """Visualize self-attention mechanism"""
    print("\n" + "="*50)
    print("Self-Attention Visualization")
    print("="*50)
    
    # 创建一个简单的自注意力层
    d_model = 512 # 模型维度
    num_heads = 8 # 头数
    attention = MultiHeadAttention(d_model, num_heads) # 创建一个自注意力层
    
    # 创建一个随机序列
    batch_size = 1 # 批量大小
    seq_length = 10 # 序列长度
    x = torch.randn(batch_size, seq_length, d_model) # 创建一个形状为(batch_size, seq_length, d_model)的随机序列
    
    # 计算自注意力
    _, attention_weights = attention(x, x, x) # 计算自注意力
    
    # 可视化第一个头的注意力权重
    plt.figure(figsize=(10, 8))
    plt.imshow(attention_weights[0, 0].detach().numpy(), cmap='viridis') # 可视化第一个头的注意力权重
    plt.xlabel('Key Position') # 设置x轴标签
    plt.ylabel('Query Position') # 设置y轴标签
    plt.title('Attention Weights of First Head') # 设置标题
    plt.colorbar() # 显示颜色条
    plt.show() # 显示图像


# 演示Transformer的核心功能
def demonstrate_transformer():
    """Demonstrate core functionality of Transformer"""
    print("\n" + "="*50)
    print("Transformer Demonstration")
    print("="*50)
    
    # 创建一个小型Transformer
    src_vocab_size = 5000 # 源词汇表大小
    tgt_vocab_size = 5000 # 目标词汇表大小
    d_model = 128 # 模型维度
    num_heads = 4 # 头数
    num_layers = 2 # 层数
    d_ff = 512 # 前馈神经网络的中间维度
    
    transformer = Transformer(
        src_vocab_size, 
        tgt_vocab_size, 
        d_model, 
        num_heads, 
        num_layers, 
        d_ff
    ).to(device) # 将输入的transformer张量通过设备
    
    # 打印模型结构
    print(transformer)
    
    # 创建一些随机输入
    batch_size = 2 # 批量大小
    src_seq_length = 10 # 源序列长度
    tgt_seq_length = 8 # 目标序列长度
    
    src = torch.randint(1, src_vocab_size, (batch_size, src_seq_length)).to(device) # 创建一个形状为(batch_size, src_seq_length)的随机序列
    tgt = torch.randint(1, tgt_vocab_size, (batch_size, tgt_seq_length)).to(device) # 创建一个形状为(batch_size, tgt_seq_length)的随机序列
    
    # 创建掩码
    src_mask, tgt_mask = create_masks(src, tgt) # 创建源序列和目标序列的掩码
    
    # 前向传播
    print("\n执行前向传播...")
    output = transformer(src, tgt, src_mask, tgt_mask) # 将输入的src张量和tgt张量通过Transformer模型    
    
    print(f"输入源序列形状: {src.shape}") # 打印输入源序列的形状
    print(f"输入目标序列形状: {tgt.shape}") # 打印输入目标序列的形状
    print(f"输出序列形状: {output.shape}") # 打印输出序列的形状
    print(f"输出第一个样本的第一个时间步的前5个logits: {output[0, 0, :5]}") # 打印输出第一个样本的第一个时间步的前5个logits


# 构建一个简单的翻译数据集
class SimpleTranslationDataset(Dataset):
    def __init__(self, size=1000, max_length=10):
        """
        创建一个简单的数学表达式翻译数据集
        英文：'1 plus 2 equals 3'
        中文：'1 加 2 等于 3'
        
        参数:
            size: 数据集大小
            max_length: 最大序列长度
        """
        self.size = size # 数据集大小
        self.max_length = max_length # 最大序列长度
        
        # 创建词汇表
        self.en_vocab = {
            '<pad>': 0, '<sos>': 1, '<eos>': 2,
            '0': 3, '1': 4, '2': 5, '3': 6, '4': 7, 
            '5': 8, '6': 9, '7': 10, '8': 11, '9': 12,
            'plus': 13, 'minus': 14, 'times': 15, 
            'divided': 16, 'by': 17, 'equals': 18
        }
        
        self.cn_vocab = {
            '<pad>': 0, '<sos>': 1, '<eos>': 2,
            '0': 3, '1': 4, '2': 5, '3': 6, '4': 7, 
            '5': 8, '6': 9, '7': 10, '8': 11, '9': 12,
            '加': 13, '减': 14, '乘': 15, '除以': 16, '等于': 17
        }
        # 创建一个字典，将词汇表中的索引映射为词汇
        self.en_i2w = {i: w for w, i in self.en_vocab.items()} # 将词汇表中的索引映射为词汇
        self.cn_i2w = {i: w for w, i in self.cn_vocab.items()} # 将词汇表中的索引映射为词汇
        
        # 生成数据
        self.data = self._generate_data() # 生成数据
    
    def _generate_data(self):
        """生成数学表达式数据集"""
        data = []
        operations = [
            ('plus', '加', lambda x, y: x + y), # 加法操作
            ('minus', '减', lambda x, y: x - y), # 减法操作
            ('times', '乘', lambda x, y: x * y), # 乘法操作
            ('divided by', '除以', lambda x, y: x / y if y != 0 else 0) # 除法操作
        ]
        
        for _ in range(self.size):
            # 生成随机数
            a = np.random.randint(0, 10) # 生成一个0到10之间的随机整数
            b = np.random.randint(1, 10)  # 避免除以0
            
            # 选择随机操作
            op_i = np.random.randint(0, len(operations)) # 生成一个0到len(operations)之间的随机整数
            en_op, cn_op, func = operations[op_i] # 选择随机操作
            
            # 计算结果
            result = round(func(a, b)) # 计算结果
            
            # 构建表达式
            if op_i == 3:  # 除法特殊处理
                en_expr = f"{a} divided by {b} equals {result}".split() # 构建表达式
            else:
                en_expr = f"{a} {en_op} {b} equals {result}".split() # 构建表达式
            
            cn_expr = f"{a} {cn_op} {b} 等于 {result}".split() # 构建表达式
            
            # 转换为索引
            en_indices = [self.en_vocab['<sos>']] # 初始化索引列表
            for word in en_expr:
                if word in self.en_vocab:
                    en_indices.append(self.en_vocab[word]) # 将词汇添加到索引列表中
                else:
                    # 处理复合词
                    for w in word.split():
                        if w in self.en_vocab:
                            en_indices.append(self.en_vocab[w]) # 将词汇添加到索引列表中
            en_indices.append(self.en_vocab['<eos>']) # 将结束符号添加到索引列表中
            
            cn_indices = [self.cn_vocab['<sos>']] # 初始化索引列表
            for word in cn_expr:
                if word in self.cn_vocab:
                    cn_indices.append(self.cn_vocab[word]) # 将词汇添加到索引列表中
            cn_indices.append(self.cn_vocab['<eos>']) # 将结束符号添加到索引列表中
            
            # 填充到最大长度
            en_indices = en_indices[:self.max_length] # 将索引列表填充到最大长度
            cn_indices = cn_indices[:self.max_length] # 将索引列表填充到最大长度
            
            en_indices = en_indices + [self.en_vocab['<pad>']] * (self.max_length - len(en_indices)) # 将索引列表填充到最大长度
            cn_indices = cn_indices + [self.cn_vocab['<pad>']] * (self.max_length - len(cn_indices)) # 将索引列表填充到最大长度
            
            data.append((en_indices, cn_indices)) # 将索引列表添加到数据列表中
        
        return data
    
    def __len__(self):
        """获取数据集的大小"""
        return self.size
    
    def __getitem__(self, idx):
        """获取数据集中的一个样本"""
        return torch.tensor(self.data[idx][0]), torch.tensor(self.data[idx][1])


# 训练Transformer模型
def train_simple_transformer():
    """Train a simple Transformer translation model"""
    print("\n" + "="*50)
    print("Training Simple Translation Task")
    print("="*50)
    
    # 创建数据集和数据加载器
    dataset = SimpleTranslationDataset() # 创建数据集
    train_size = int(0.8 * len(dataset)) # 训练集大小
    test_size = len(dataset) - train_size # 测试集大小
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size]) # 将数据集分割为训练集和测试集
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True) # 创建训练数据加载器
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False) # 创建测试数据加载器
    
    print(f"训练集大小: {len(train_dataset)}") # 打印训练集大小
    print(f"测试集大小: {len(test_dataset)}") # 打印测试集大小
    print(f"英文词汇量: {len(dataset.en_vocab)}") # 打印英文词汇量
    print(f"中文词汇量: {len(dataset.cn_vocab)}") # 打印中文词汇量
    
    # 创建模型
    src_vocab_size = len(dataset.en_vocab) # 源词汇表大小
    tgt_vocab_size = len(dataset.cn_vocab) # 目标词汇表大小
    d_model = 128 # 模型维度
    num_heads = 4 # 头数
    num_layers = 2 # 层数
    d_ff = 512 # 前馈神经网络的中间维度
    
    model = Transformer(
        src_vocab_size, 
        tgt_vocab_size, 
        d_model, 
        num_heads, 
        num_layers, 
        d_ff
    ).to(device) # 将输入的model张量通过设备
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略<pad>的损失
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9) # 定义优化器
    
    # 训练循环
    epochs = 5 # 训练轮数
    train_losses = [] # 初始化训练损失列表
    test_losses = [] # 初始化测试损失列表
    
    for epoch in range(1, epochs+1):
        model.train() # 设置模型为训练模式
        epoch_loss = 0 # 初始化损失为0
        
        for i, (src, tgt) in enumerate(train_loader):
            src, tgt = src.to(device), tgt.to(device) # 将输入的src张量和tgt张量通过设备
            
            # 目标序列的输入和输出
            tgt_input = tgt[:, :-1]  # 移除最后一个token
            tgt_output = tgt[:, 1:]  # 移除第一个token <sos>
            
            # 创建掩码
            src_mask, tgt_mask = create_masks(src, tgt_input) # 创建源序列和目标序列的掩码
            
            # 前向传播
            predictions = model(src, tgt_input, src_mask, tgt_mask) # 将输入的src张量和tgt_input张量通过Transformer模型
            
            # 重塑张量用于计算损失
            predictions = predictions.contiguous().view(-1, predictions.size(-1)) # 将输入的predictions张量展平
            tgt_output = tgt_output.contiguous().view(-1) # 将输入的tgt_output张量展平
            
            # 计算损失
            loss = criterion(predictions, tgt_output) # 计算损失
            
            # 反向传播和优化
            optimizer.zero_grad() # 将优化器的梯度清零
            loss.backward() # 反向传播
            optimizer.step() # 更新模型参数
            
            epoch_loss += loss.item() # 将损失添加到epoch_loss中
            
            if (i+1) % 10 == 0:
                print(f"Epoch {epoch}/{epochs}, Batch {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}") # 打印训练损失
        
        avg_train_loss = epoch_loss / len(train_loader) # 计算平均损失
        train_losses.append(avg_train_loss) # 将平均损失添加到train_losses中
        
        # 在测试集上评估
        model.eval()
        test_loss = 0
        
        with torch.no_grad():
            for src, tgt in test_loader:
                src, tgt = src.to(device), tgt.to(device) # 将输入的src张量和tgt张量通过设备
                
                tgt_input = tgt[:, :-1] # 移除最后一个token
                tgt_output = tgt[:, 1:] # 移除第一个token <sos>
                
                src_mask, tgt_mask = create_masks(src, tgt_input)   # 创建源序列和目标序列的掩码
                
                predictions = model(src, tgt_input, src_mask, tgt_mask) # 将输入的src张量和tgt_input张量通过Transformer模型
                
                predictions = predictions.contiguous().view(-1, predictions.size(-1)) # 将输入的predictions张量展平
                tgt_output = tgt_output.contiguous().view(-1) # 将输入的tgt_output张量展平
                
                loss = criterion(predictions, tgt_output) # 计算损失
                test_loss += loss.item() # 将损失添加到test_loss中
        
        avg_test_loss = test_loss / len(test_loader) # 计算平均损失
        test_losses.append(avg_test_loss) # 将平均损失添加到test_losses中
        
        print(f"Epoch {epoch}/{epochs}, Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}") # 打印训练和测试损失
    
    # 可视化训练过程
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs+1), train_losses, label='Training Loss') # 绘制训练损失
    plt.plot(range(1, epochs+1), test_losses, label='Test Loss') # 绘制测试损失
    plt.xlabel('Epoch') # 设置x轴标签
    plt.ylabel('Loss') # 设置y轴标签
    plt.title('Training and Test Loss') # 设置标题
    plt.legend() # 显示图例
    plt.savefig('transformer_training.png') # 保存图像
    print("Training visualization saved as transformer_training.png") # 打印训练可视化保存路径
    
    return model, dataset


# 展示翻译结果
def show_translation_examples(model, dataset, num_examples=5):
    """Show translation examples"""
    print("\n" + "="*50)
    print("Translation Examples")
    print("="*50)
    
    model.eval()
    
    # 创建一个测试数据加载器
    test_loader = DataLoader(dataset, batch_size=1, shuffle=True) # 创建一个测试数据加载器
    data_iter = iter(test_loader) # 创建一个数据迭代器
    
    with torch.no_grad():
        for _ in range(num_examples):
            src, tgt = next(data_iter) # 获取下一个样本
            src, tgt = src.to(device), tgt.to(device) # 将输入的src张量和tgt张量通过设备
            
            # 准备输入
            tgt_input = tgt[:, :-1] # 移除最后一个token
            
            # 创建掩码
            src_mask, tgt_mask = create_masks(src, tgt_input) # 创建源序列和目标序列的掩码
            
            # 获取预测
            predictions = model(src, tgt_input, src_mask, tgt_mask) # 将输入的src张量和tgt_input张量通过Transformer模型
            _, predicted = torch.max(predictions, dim=2) # 获取预测
            
            # 转换为文本
            src_text = ' '.join([dataset.en_i2w[idx.item()] for idx in src[0] if idx.item() > 0]) # 将源文本转换为文本
            tgt_text = ' '.join([dataset.cn_i2w[idx.item()] for idx in tgt[0] if idx.item() > 0]) # 将目标文本转换为文本    
            pred_text = ' '.join([dataset.cn_i2w[idx.item()] for idx in predicted[0] if idx.item() > 0]) # 将预测文本转换为文本
            
            print(f"源文本: {src_text}") # 打印源文本
            print(f"目标翻译: {tgt_text}") # 打印目标翻译
            print(f"模型翻译: {pred_text}") # 打印模型翻译
            print('-' * 50) # 打印分隔符


def main():
    """Main function"""
    print("\n" + "*"*70)
    print("*" + " "*25 + "Transformer Implementation" + " "*24 + "*")
    print("*"*70)
    
    # 可视化位置编码
    visualize_positional_encoding()
    
    # 可视化自注意力
    visualize_self_attention()
    
    # 演示Transformer模型
    demonstrate_transformer()
    
    # 训练简单翻译模型
    model, dataset = train_simple_transformer()
    
    # 显示翻译示例
    show_translation_examples(model, dataset)
    
    print("\n" + "*"*70)
    print("*" + " "*23 + "Transformer Implementation Complete" + " "*22 + "*")
    print("*"*70)


if __name__ == "__main__":
    main()