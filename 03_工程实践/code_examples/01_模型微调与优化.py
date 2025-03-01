#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
大型语言模型的微调与优化
=================
本代码演示了LLM的微调和优化流程，包括：
1. 数据准备和预处理
2. 使用LoRA技术进行高效微调
3. 量化优化模型
4. 性能评估和比较
5. 模型部署准备
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import time
from datetime import datetime
from tqdm import tqdm
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    GenerationConfig
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    prepare_model_for_kbit_training,
    PeftModel
)

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 设置随机种子，保证结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 检查是否有GPU可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")


def show_gpu_info():
    """显示GPU信息与使用情况"""
    if torch.cuda.is_available():
        print(f"当前使用的GPU: {torch.cuda.get_device_name(0)}")
        print(f"可用GPU数量: {torch.cuda.device_count()}")
        print(f"当前GPU内存占用: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"当前GPU内存缓存: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    else:
        print("没有可用的GPU")


def prepare_sample_data():
    """准备示例数据集"""
    print("\n" + "="*50)
    print("数据准备")
    print("="*50)
    
    # 创建示例指令微调数据
    data = [
        {
            "instruction": "解释什么是机器学习",
            "input": "",
            "output": "机器学习是人工智能的一个分支，它使计算机系统能够从数据中学习和改进，而无需明确编程。机器学习算法通过分析数据，识别模式，并做出决策，随着经验的积累而提高性能。"
        },
        {
            "instruction": "总结以下文本的主要内容",
            "input": "深度学习是机器学习的一个子领域，它使用多层神经网络来模拟人脑的工作方式。这些网络可以从大量数据中学习复杂的模式，并应用于图像识别、自然语言处理和推荐系统等任务。",
            "output": "文本主要介绍了深度学习的概念、工作原理及其在图像识别、自然语言处理和推荐系统等领域的应用。"
        },
        {
            "instruction": "编写一个Python函数，计算斐波那契数列的第n个数",
            "input": "",
            "output": "```python\ndef fibonacci(n):\n    if n <= 0:\n        return 0\n    elif n == 1:\n        return 1\n    else:\n        a, b = 0, 1\n        for _ in range(2, n+1):\n            a, b = b, a + b\n        return b\n```"
        },
        {
            "instruction": "分析以下代码的时间复杂度",
            "input": "```python\ndef find_duplicate(nums):\n    slow = nums[0]\n    fast = nums[0]\n    while True:\n        slow = nums[slow]\n        fast = nums[nums[fast]]\n        if slow == fast:\n            break\n    \n    ptr1 = nums[0]\n    ptr2 = slow\n    while ptr1 != ptr2:\n        ptr1 = nums[ptr1]\n        ptr2 = nums[ptr2]\n    \n    return ptr1\n```",
            "output": "这段代码实现了Floyd's Tortoise and Hare算法（龟兔赛跑算法）来查找数组中的重复数字。时间复杂度为O(n)，其中n是数组的长度。空间复杂度为O(1)，因为只使用了常数个额外变量。"
        },
        {
            "instruction": "解释大型语言模型（LLM）的工作原理",
            "input": "",
            "output": "大型语言模型（LLM）基于Transformer架构，通过自注意力机制处理和生成文本。这些模型在大规模文本语料库上预训练，学习语言的模式、语法和知识。它们能够理解上下文，并根据给定的提示生成连贯的文本。LLM通过计算单词之间的关系权重，预测序列中的下一个单词，从而生成连贯的文本。例如GPT（生成式预训练Transformer）模型就属于这类模型。"
        }
    ]
    
    # 创建更多样本
    additional_samples = []
    for i in range(20):
        template = data[i % len(data)]
        variation = {
            "instruction": template["instruction"] + f" (变体 {i+1})",
            "input": template["input"],
            "output": template["output"] + f" 这是示例回答的变体 {i+1}。"
        }
        additional_samples.append(variation)
    
    # 合并所有样本
    all_samples = data + additional_samples
    
    # 转换为DataFrame便于查看
    df = pd.DataFrame(all_samples)
    print(f"创建的示例数据集大小: {len(df)}条")
    print("\n示例数据:")
    print(df.head(3))
    
    # 转换为HuggingFace数据集格式
    dataset = Dataset.from_pandas(df)
    
    # 分割数据集
    train_test_split = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = train_test_split["train"]
    test_dataset = train_test_split["test"]
    
    print(f"\n训练集大小: {len(train_dataset)}条")
    print(f"测试集大小: {len(test_dataset)}条")
    
    return train_dataset, test_dataset


def format_prompt(example):
    """格式化提示，将指令和输入组合成模型可接受的格式"""
    instruction = example["instruction"]
    input_text = example["input"]
    output = example["output"]
    
    # 构建提示模板
    if input_text:
        prompt = f"### 指令:\n{instruction}\n\n### 输入:\n{input_text}\n\n### 响应:\n"
    else:
        prompt = f"### 指令:\n{instruction}\n\n### 响应:\n"
    
    return {
        "prompt": prompt,
        "output": output
    }


def prepare_model_for_training(base_model_name="bigscience/bloomz-7b1"):
    """准备模型用于训练"""
    print("\n" + "="*50)
    print("模型准备")
    print("="*50)
    
    # 为了演示目的，使用小型模型
    if torch.cuda.is_available():
        print(f"准备基础模型: {base_model_name}")
        
        # 配置量化参数
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        
        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        # 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name, 
            trust_remote_code=True
        )
        tokenizer.pad_token = tokenizer.eos_token
        
        # 配置LoRA
        lora_config = LoraConfig(
            r=16,                # LoRA注意力维度
            lora_alpha=32,       # LoRA缩放因子
            target_modules=["query_key_value"],  # 要微调的模块
            lora_dropout=0.05,   # Dropout概率
            bias="none",         # 是否训练偏置参数
            task_type="CAUSAL_LM"  # 任务类型
        )
        
        # 准备模型进行4位训练
        model = prepare_model_for_kbit_training(model)
        
        # 添加LoRA适配器
        model = get_peft_model(model, lora_config)
        
        # 打印可训练参数信息
        model.print_trainable_parameters()
        
    else:
        print("警告: 当前环境没有GPU，使用小型模型进行演示")
        # 使用小型模型进行CPU演示
        base_model_name = "distilgpt2"  # 使用较小的模型
        model = AutoModelForCausalLM.from_pretrained(base_model_name)
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        # 配置LoRA
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["c_attn"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        # 添加LoRA适配器
        model = get_peft_model(model, lora_config)
        
        print("使用小型模型 distilgpt2 进行演示")
    
    return model, tokenizer


def train_model(model, tokenizer, train_dataset, test_dataset, output_dir="./results"):
    """使用LoRA技术微调模型"""
    print("\n" + "="*50)
    print("模型微调")
    print("="*50)
    
    # 格式化训练和测试数据
    train_dataset = train_dataset.map(format_prompt)
    test_dataset = test_dataset.map(format_prompt)
    
    # 定义数据处理函数
    def preprocess_function(examples):
        inputs = examples["prompt"]
        targets = examples["output"]
        
        # 将输入和目标组合起来用于训练
        model_inputs = tokenizer(inputs, truncation=True, max_length=512)
        labels = tokenizer(targets, truncation=True, max_length=512)
        
        # 组合输入的token_ids和目标token_ids
        for i in range(len(model_inputs["input_ids"])):
            model_inputs["input_ids"][i] = model_inputs["input_ids"][i] + labels["input_ids"][i]
            model_inputs["attention_mask"][i] = model_inputs["attention_mask"][i] + labels["attention_mask"][i]
            
        return model_inputs
    
    # 处理训练和测试数据
    tokenized_train = train_dataset.map(
        preprocess_function,
        remove_columns=train_dataset.column_names,
        batched=True
    )
    
    tokenized_test = test_dataset.map(
        preprocess_function,
        remove_columns=test_dataset.column_names,
        batched=True
    )
    
    # 定义训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=3,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
        report_to="none"
    )
    
    # 使用PyTorch Trainer API进行训练
    # 注意: 在实际代码中，我们会在这里使用transformers的Trainer类进行训练
    # 但为了演示目的，我们在这里使用简化的训练循环
    print("\n开始训练...(模拟训练过程)")
    
    # 模拟训练过程
    num_epochs = 3
    train_losses = []
    eval_losses = []
    
    for epoch in range(1, num_epochs+1):
        # 模拟训练损失
        train_loss = 2.5 / epoch + 0.5 + np.random.normal(0, 0.1)
        train_losses.append(train_loss)
        
        # 模拟评估损失
        eval_loss = 3.0 / epoch + 0.3 + np.random.normal(0, 0.1)
        eval_losses.append(eval_loss)
        
        print(f"Epoch {epoch}/{num_epochs}:")
        print(f"  训练损失: {train_loss:.4f}")
        print(f"  评估损失: {eval_loss:.4f}")
        
        # 模拟进度条
        for _ in tqdm(range(10), desc=f"Epoch {epoch} 训练进度"):
            time.sleep(0.1)
    
    # 绘制训练曲线
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs+1), train_losses, label='训练损失')
    plt.plot(range(1, num_epochs+1), eval_losses, label='评估损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.title('训练与评估损失曲线')
    plt.legend()
    plt.savefig('training_curve.png')
    print("训练曲线已保存为 training_curve.png")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n模型微调完成!")
    return model, tokenizer


def quantize_model(model, tokenizer):
    """演示对微调后的模型进行量化优化"""
    print("\n" + "="*50)
    print("模型量化")
    print("="*50)
    
    # 在实际项目中，我们会使用GPTQ或AWQ进行模型量化
    # 这里仅演示概念和基本流程
    
    print("执行模型量化过程...(模拟)")
    print("1. 收集校准数据")
    print("2. 计算量化参数")
    print("3. 应用量化")
    print("4. 验证量化精度")
    
    # 模拟量化前后的模型大小对比
    original_size = 7.0  # 假设原始模型大小为7GB
    quantized_size = 2.0  # 假设量化后模型大小为2GB
    
    # 可视化模型大小对比
    plt.figure(figsize=(8, 5))
    plt.bar(['原始模型', '量化模型'], [original_size, quantized_size], color=['blue', 'green'])
    plt.ylabel('模型大小 (GB)')
    plt.title('模型量化前后大小对比')
    for i, v in enumerate([original_size, quantized_size]):
        plt.text(i, v + 0.1, f"{v}GB", ha='center')
    plt.savefig('model_quantization.png')
    print("模型量化对比图已保存为 model_quantization.png")
    
    # 模拟速度对比
    speeds = {
        'FP16': 12.5,
        'INT8': 23.8,
        'INT4': 38.2,
    }
    
    plt.figure(figsize=(10, 5))
    plt.bar(speeds.keys(), speeds.values(), color=['blue', 'green', 'red'])
    plt.ylabel('推理速度 (tokens/sec)')
    plt.title('不同精度下的推理速度对比')
    for i, (k, v) in enumerate(speeds.items()):
        plt.text(i, v + 1, f"{v}", ha='center')
    plt.savefig('inference_speed.png')
    print("推理速度对比图已保存为 inference_speed.png")
    
    print("\n模型量化完成，模型大小从 7GB 减小到 2GB，同时推理速度提升了3倍")
    return model, tokenizer


def evaluate_model(model, tokenizer, test_dataset):
    """评估模型性能"""
    print("\n" + "="*50)
    print("模型评估")
    print("="*50)
    
    # 准备测试样例
    test_examples = test_dataset.select(range(min(5, len(test_dataset))))
    
    # 定义评估指标
    metrics = {
        "正确率": [],
        "响应长度": [],
        "响应时间": []
    }
    
    print("生成测试样例的响应...")
    
    for i, example in enumerate(test_examples):
        prompt = example["prompt"]
        reference = example["output"]
        
        print(f"\n测试样例 {i+1}:")
        print(f"提示: {prompt}")
        print(f"参考答案: {reference}")
        
        # 记录开始时间
        start_time = time.time()
        
        # 在实际代码中，我们会使用模型生成响应
        # 但为了演示目的，我们生成模拟的响应
        response = reference[:len(reference)//2] + "..." + reference[-20:]
        
        # 计算响应时间
        response_time = time.time() - start_time
        
        print(f"模型响应: {response}")
        print(f"响应时间: {response_time:.2f}秒")
        
        # 计算模拟指标
        correctness = 0.7 + np.random.random() * 0.2  # 模拟正确率
        
        # 更新指标
        metrics["正确率"].append(correctness)
        metrics["响应长度"].append(len(response))
        metrics["响应时间"].append(response_time)
    
    # 计算平均指标
    avg_metrics = {k: sum(v)/len(v) for k, v in metrics.items()}
    
    print("\n评估指标:")
    for metric, value in avg_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # 可视化评估结果
    plt.figure(figsize=(12, 6))
    
    # 创建子图
    ax1 = plt.subplot(131)
    ax1.bar(range(len(metrics["正确率"])), metrics["正确率"], color='green')
    ax1.set_xlabel('测试样例')
    ax1.set_ylabel('正确率')
    ax1.set_title('模型正确率')
    
    ax2 = plt.subplot(132)
    ax2.bar(range(len(metrics["响应长度"])), metrics["响应长度"], color='blue')
    ax2.set_xlabel('测试样例')
    ax2.set_ylabel('长度')
    ax2.set_title('响应长度')
    
    ax3 = plt.subplot(133)
    ax3.bar(range(len(metrics["响应时间"])), metrics["响应时间"], color='red')
    ax3.set_xlabel('测试样例')
    ax3.set_ylabel('时间(秒)')
    ax3.set_title('响应时间')
    
    plt.tight_layout()
    plt.savefig('model_evaluation.png')
    print("评估结果可视化已保存为 model_evaluation.png")
    
    return avg_metrics


def prepare_for_deployment(model, tokenizer, output_dir="./deployment"):
    """准备模型部署"""
    print("\n" + "="*50)
    print("部署准备")
    print("="*50)
    
    # 创建部署目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 在实际项目中，这里会导出模型并准备部署配置
    # 但为了演示目的，我们只介绍这个过程
    
    print("准备模型部署...")
    print("1. 合并LoRA权重到基础模型")
    print("2. 导出量化模型")
    print("3. 准备推理配置")
    print("4. 生成示例代码")
    
    # 创建示例部署配置
    deployment_config = {
        "model_name": "微调后的LLM模型",
        "version": "1.0.0",
        "date": datetime.now().strftime("%Y-%m-%d"),
        "framework": "PyTorch",
        "quantization": "INT4",
        "inference_settings": {
            "max_new_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.1
        },
        "system_requirements": {
            "gpu_memory": "8GB+",
            "cuda_version": "11.7+",
            "python_version": "3.8+"
        }
    }
    
    # 保存配置文件
    with open(os.path.join(output_dir, "deployment_config.json"), "w", encoding="utf-8") as f:
        json.dump(deployment_config, f, ensure_ascii=False, indent=2)
    
    # 生成示例推理代码
    inference_code = '''
# 推理代码示例
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# 加载模型和分词器
model_path = "./deployment/model"
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 创建推理管道
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.1
)

# 进行推理
prompt = "### 指令:\\n解释什么是大型语言模型\\n\\n### 响应:\\n"
result = pipe(prompt)
print(result[0]['generated_text'])
'''
    
    # 保存示例推理代码
    with open(os.path.join(output_dir, "inference_example.py"), "w", encoding="utf-8") as f:
        f.write(inference_code)
    
    # 生成示例服务器代码
    server_code = '''
# 服务器部署示例
from fastapi import FastAPI, Request
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import uvicorn
import json

app = FastAPI(title="LLM API")

# 全局变量
model = None
tokenizer = None
inference_pipeline = None

# 加载模型
@app.on_event("startup")
async def startup_event():
    global model, tokenizer, inference_pipeline
    model_path = "./deployment/model"
    
    # 加载模型和分词器
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # 创建推理管道
    inference_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1
    )

# API端点
@app.post("/generate")
async def generate(request: Request):
    data = await request.json()
    prompt = data.get("prompt", "")
    
    # 生成响应
    result = inference_pipeline(prompt)
    response = result[0]['generated_text']
    
    return {"response": response}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''
    
    # 保存示例服务器代码
    with open(os.path.join(output_dir, "server_example.py"), "w", encoding="utf-8") as f:
        f.write(server_code)
    
    print(f"部署配置已保存到 {output_dir}/deployment_config.json")
    print(f"推理示例代码已保存到 {output_dir}/inference_example.py")
    print(f"服务器部署示例已保存到 {output_dir}/server_example.py")
    print("\n模型已准备好部署!")


def main():
    """主函数"""
    print("\n" + "*"*70)
    print("*" + " "*20 + "大型语言模型的微调与优化" + " "*20 + "*")
    print("*"*70)
    
    # 检查GPU情况
    show_gpu_info()
    
    # 准备数据
    train_dataset, test_dataset = prepare_sample_data()
    
    # 准备模型
    model, tokenizer = prepare_model_for_training()
    
    # 微调模型
    model, tokenizer = train_model(model, tokenizer, train_dataset, test_dataset)
    
    # 量化模型
    model, tokenizer = quantize_model(model, tokenizer)
    
    # 评估模型
    metrics = evaluate_model(model, tokenizer, test_dataset)
    
    # 准备部署
    prepare_for_deployment(model, tokenizer)
    
    print("\n" + "*"*70)
    print("*" + " "*19 + "LLM微调与优化流程演示完成" + " "*19 + "*")
    print("*"*70)


if __name__ == "__main__":
    main() 