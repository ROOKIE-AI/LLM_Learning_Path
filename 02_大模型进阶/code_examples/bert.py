import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LayerNorm(nn.Module):
    """
    层归一化
    """
    def __init__(self, hidden_size, eps=1e-12):
        """
        hidden_size: 隐藏层大小
        eps: 防止除以0
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size)) # 权重
        self.bias = nn.Parameter(torch.zeros(hidden_size)) # 偏置
        self.eps = eps # 防止除以0

    def forward(self, x):
        """
        前向传播
        x: 输入
        """
        u = x.mean(-1, keepdim=True) # 均值
        s = (x - u).pow(2).mean(-1, keepdim=True) # 方差
        x = (x - u) / torch.sqrt(s + self.eps) # 归一化
        return self.weight * x + self.bias # 缩放和平移

class MultiHeadAttention(nn.Module):
    """
    多头注意力机制
    """
    def __init__(self, hidden_size, num_attention_heads, dropout_rate=0.1):
        """
        hidden_size: 隐藏层大小
        num_attention_heads: 注意力头数
        dropout_rate: 丢弃率
        """
        super(MultiHeadAttention, self).__init__()
        self.num_attention_heads = num_attention_heads # 注意力头数
        self.attention_head_size = int(hidden_size / num_attention_heads) # 每个注意力头的维度
        self.all_head_size = self.num_attention_heads * self.attention_head_size # 所有注意力头的维度
        
        self.query = nn.Linear(hidden_size, self.all_head_size) # 查询
        self.key = nn.Linear(hidden_size, self.all_head_size) # 键
        self.value = nn.Linear(hidden_size, self.all_head_size) # 值    
        
        self.dropout = nn.Dropout(dropout_rate) # 丢弃
        self.out = nn.Linear(hidden_size, hidden_size) # 输出
        
    def transpose_for_scores(self, x):
        """
        将x变形为(batch_size, num_attention_heads, seq_length, attention_head_size)
        """
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size) # 变形
        x = x.view(*new_x_shape) # 变形
        return x.permute(0, 2, 1, 3) # 变形
    
    def forward(self, hidden_states, attention_mask=None):
        """
        前向传播
        hidden_states: 输入
        attention_mask: 注意力掩码
        """
        mixed_query_layer = self.query(hidden_states) # 查询
        mixed_key_layer = self.key(hidden_states) # 键
        mixed_value_layer = self.value(hidden_states) # 值
        
        query_layer = self.transpose_for_scores(mixed_query_layer) 
        key_layer = self.transpose_for_scores(mixed_key_layer) 
        value_layer = self.transpose_for_scores(mixed_value_layer) 
        
        # 计算注意力分数
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) # 矩阵乘法
        attention_scores = attention_scores / math.sqrt(self.attention_head_size) # 缩放
        
        # 应用注意力掩码（如果提供）
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask # 加法
        
        # 归一化注意力分数
        attention_probs = nn.Softmax(dim=-1)(attention_scores) # 归一化
        attention_probs = self.dropout(attention_probs) # 丢弃
        
        context_layer = torch.matmul(attention_probs, value_layer) # 矩阵乘法
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous() 
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,) 
        context_layer = context_layer.view(*new_context_layer_shape) 
        
        output = self.out(context_layer) # 输出
        return output

class PositionwiseFeedForward(nn.Module):
    """
    位置前馈神经网络
    """
    def __init__(self, hidden_size, intermediate_size, dropout_rate=0.1):
        """
        hidden_size: 隐藏层大小
        intermediate_size: 中间层大小
        dropout_rate: 丢弃率
        """
        super(PositionwiseFeedForward, self).__init__()
        self.dense1 = nn.Linear(hidden_size, intermediate_size) # 线性层
        self.dense2 = nn.Linear(intermediate_size, hidden_size) # 线性层
        self.dropout = nn.Dropout(dropout_rate) # 丢弃
        
    def forward(self, hidden_states):
        """
        前向传播
        hidden_states: 输入
        """
        hidden_states = self.dense1(hidden_states) # 线性层
        hidden_states = F.gelu(hidden_states) # 激活函数
        hidden_states = self.dense2(hidden_states) # 线性层
        hidden_states = self.dropout(hidden_states) # 丢弃
        return hidden_states

class TransformerBlock(nn.Module):
    """
    变压器块
    """
    def __init__(self, hidden_size, num_attention_heads, intermediate_size, dropout_rate=0.1):
        """
        hidden_size: 隐藏层大小
        num_attention_heads: 注意力头数
        intermediate_size: 中间层大小
        dropout_rate: 丢弃率
        """
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(hidden_size, num_attention_heads, dropout_rate) # 多头注意力机制
        self.attention_norm = LayerNorm(hidden_size) # 层归一化
        self.ffn = PositionwiseFeedForward(hidden_size, intermediate_size, dropout_rate) # 位置前馈神经网络
        self.ffn_norm = LayerNorm(hidden_size) # 层归一化
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, hidden_states, attention_mask=None):
        """
        前向传播
        hidden_states: 输入
        attention_mask: 注意力掩码
        """
        # 自注意力机制
        attention_output = self.attention(hidden_states, attention_mask) # 多头注意力机制
        attention_output = self.dropout(attention_output) # 丢弃
        attention_output = self.attention_norm(attention_output + hidden_states) # 层归一化
        
        # 前馈神经网络
        ffn_output = self.ffn(attention_output) # 位置前馈神经网络
        ffn_output = self.ffn_norm(ffn_output + attention_output) # 层归一化
        
        return ffn_output

class BertEmbeddings(nn.Module):
    """
    BERT词嵌入层
    """
    def __init__(self, vocab_size, hidden_size, max_position_embeddings, type_vocab_size, dropout_rate=0.1):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size) # 词嵌入层
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size) # 位置嵌入层
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size) # 类型嵌入层
        
        self.layer_norm = LayerNorm(hidden_size) # 层归一化
        self.dropout = nn.Dropout(dropout_rate) # 丢弃
        
    def forward(self, input_ids, token_type_ids=None):
        """
        前向传播
        input_ids: 输入
        token_type_ids: 类型嵌入
        """
        seq_length = input_ids.size(1) # 序列长度
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device) # 位置id
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids) # 扩展
        
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids) # 类型嵌入
        
        words_embeddings = self.word_embeddings(input_ids) # 词嵌入
        position_embeddings = self.position_embeddings(position_ids) # 位置嵌入
        token_type_embeddings = self.token_type_embeddings(token_type_ids) # 类型嵌入
        
        embeddings = words_embeddings + position_embeddings + token_type_embeddings # 嵌入
        embeddings = self.layer_norm(embeddings) # 层归一化
        embeddings = self.dropout(embeddings) # 丢弃
        
        return embeddings

class BertConfig:
    """
    BERT配置
    """
    def __init__(self, vocab_size, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, 
                 intermediate_size=3072, max_position_embeddings=512, type_vocab_size=2, dropout_rate=0.1):
        """
        vocab_size: 词汇表大小
        hidden_size: 隐藏层大小
        num_hidden_layers: 隐藏层数量
        num_attention_heads: 注意力头数
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.dropout_rate = dropout_rate

class BERT(nn.Module):
    """
    BERT模型
    """
    def __init__(self, vocab_size=None, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, 
                 intermediate_size=3072, max_position_embeddings=512, type_vocab_size=2, dropout_rate=0.1):
        """
        vocab_size: 词汇表大小
        hidden_size: 隐藏层大小
        num_hidden_layers: 隐藏层数量
        num_attention_heads: 注意力头数
        intermediate_size: 中间层大小
        max_position_embeddings: 最大位置嵌入
        type_vocab_size: 类型嵌入大小
        dropout_rate: 丢弃率
        """
        super(BERT, self).__init__()
        
        # 检查是否传入了BertConfig对象
        if isinstance(vocab_size, BertConfig):
            config = vocab_size
            vocab_size = config.vocab_size
            hidden_size = config.hidden_size
            num_hidden_layers = config.num_hidden_layers
            num_attention_heads = config.num_attention_heads
            intermediate_size = config.intermediate_size
            max_position_embeddings = config.max_position_embeddings
            type_vocab_size = config.type_vocab_size
            dropout_rate = config.dropout_rate
        
        self.embeddings = BertEmbeddings(vocab_size, hidden_size, max_position_embeddings, type_vocab_size, dropout_rate) # 词嵌入层
        self.encoder = nn.ModuleList([
            TransformerBlock(hidden_size, num_attention_heads, intermediate_size, dropout_rate) # 变压器块
            for _ in range(num_hidden_layers)
        ])
        
        # 初始化权重
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """
        初始化权重
        module: 模块
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02) # 正态分布
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_() # 偏置
    
    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        """
        前向传播
        input_ids: 输入
        token_type_ids: 类型嵌入
        attention_mask: 注意力掩码
        """
        if attention_mask is not None:
            # 扩展注意力掩码 [batch_size, seq_length] -> [batch_size, 1, 1, seq_length]
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) # 扩展
            # 将0转换为-10000.0，将1保持不变
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        else:
            extended_attention_mask = None
        
        embedding_output = self.embeddings(input_ids, token_type_ids) # 词嵌入
        hidden_states = embedding_output # 隐藏状态
        
        for layer in self.encoder:
            hidden_states = layer(hidden_states, extended_attention_mask) # 变压器块
        
        return hidden_states

class BertForMaskedLM(nn.Module):
    """
    BERT用于掩码语言模型
    """
    def __init__(self, bert, vocab_size, hidden_size=768):
        """
        bert: BERT模型
        vocab_size: 词汇表大小
        hidden_size: 隐藏层大小
        """
        super(BertForMaskedLM, self).__init__()
        self.bert = bert # BERT模型
        self.cls = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), # 线性层
            nn.GELU(), # 激活函数
            LayerNorm(hidden_size), # 层归一化
            nn.Linear(hidden_size, vocab_size) # 线性层
        )

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        """
        前向传播
        input_ids: 输入
        token_type_ids: 类型嵌入
        attention_mask: 注意力掩码
        """
        sequence_output = self.bert(input_ids, token_type_ids, attention_mask) # 词嵌入
        prediction_scores = self.cls(sequence_output) # 线性层
        return prediction_scores

class BertForSequenceClassification(nn.Module):
    """
    BERT用于序列分类
    """
    def __init__(self, bert, num_labels, hidden_size=768, dropout_rate=0.1):
        """
        bert: BERT模型
        num_labels: 标签数量
        hidden_size: 隐藏层大小
        dropout_rate: 丢弃率
        """
        super(BertForSequenceClassification, self).__init__()
        self.bert = bert # BERT模型
        self.dropout = nn.Dropout(dropout_rate) # 丢弃
        self.classifier = nn.Linear(hidden_size, num_labels) # 线性层
        
    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        """
        前向传播
        input_ids: 输入
        token_type_ids: 类型嵌入
        attention_mask: 注意力掩码
        """
        sequence_output = self.bert(input_ids, token_type_ids, attention_mask) # 词嵌入
        pooled_output = sequence_output[:, 0] # 池化
        pooled_output = self.dropout(pooled_output) # 丢弃
        logits = self.classifier(pooled_output) # 线性层
        return logits

# 将上面的代码改为只在直接运行脚本时执行
if __name__ == "__main__":
    # 创建BERT配置
    config = BertConfig(vocab_size=30522, hidden_size=768, num_hidden_layers=12, num_attention_heads=12)
    
    # 创建BERT模型
    bert = BERT(config)
    
    # 打印模型信息
    print(f"BERT模型参数量: {sum(p.numel() for p in bert.parameters()):,}")
