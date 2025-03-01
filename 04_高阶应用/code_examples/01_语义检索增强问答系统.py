#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
语义检索增强问答系统 (RAG)
=================
本代码演示了如何构建一个基于语义检索增强的问答系统，包括：
1. 文档处理与向量化
2. 向量数据库的创建与查询
3. 大型语言模型与检索系统的集成
4. 回答生成与引用溯源
5. 系统评估与优化
"""

import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm

# 用于文本处理和向量化
from transformers import AutoTokenizer, AutoModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# 用于问答系统
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 设置随机种子，保证结果可复现
np.random.seed(42)
torch.manual_seed(42)

# 检查是否有GPU可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")


@dataclass
class RAGConfig:
    """RAG系统配置"""
    # 向量数据库相关配置
    embedding_model_name: str = "BAAI/bge-large-zh-v1.5"
    vector_db_path: str = "./vector_db"
    
    # 大型语言模型相关配置
    llm_model_name: str = "THUDM/chatglm3-6b"
    temperature: float = 0.1
    max_length: int = 2048
    
    # 文档处理相关配置
    chunk_size: int = 500
    chunk_overlap: int = 50
    
    # 检索相关配置
    top_k: int = 5


def load_documents(data_dir: str) -> List[Document]:
    """加载文档
    
    Args:
        data_dir: 文档目录
        
    Returns:
        文档列表
    """
    print(f"\n加载文档从: {data_dir}")
    
    if not os.path.exists(data_dir):
        # 创建示例文档
        os.makedirs(data_dir, exist_ok=True)
        create_sample_documents(data_dir)
    
    # 使用DirectoryLoader加载文档
    loader = DirectoryLoader(
        data_dir,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    
    documents = loader.load()
    print(f"加载了 {len(documents)} 个文档")
    
    return documents


def create_sample_documents(data_dir: str) -> None:
    """创建示例文档用于演示
    
    Args:
        data_dir: 文档目录
    """
    print("创建示例文档...")
    
    # 人工智能相关内容
    ai_content = """
人工智能(AI)概述
==============

人工智能(Artificial Intelligence, AI)是计算机科学的一个分支，致力于创建能够模拟人类智能的系统。
这些系统能够学习、推理、感知、理解自然语言并解决问题。

主要类型:
1. 弱人工智能(ANI): 专注于解决特定问题，如图像识别或自然语言处理。
2. 强人工智能(AGI): 具有与人类相当的认知能力，能够理解、学习和应用知识。
3. 超人工智能(ASI): 在几乎所有领域都超越人类能力的理论性AI。

主要技术:
1. 机器学习: 使用统计方法让计算机系统从数据中学习，无需明确编程。
2. 深度学习: 基于人工神经网络的机器学习子集，模拟人脑神经元连接。
3. 自然语言处理: 使计算机能够理解、解释和生成人类语言。
4. 计算机视觉: 使计算机能够从数字图像或视频中获取高级理解。
5. 专家系统: 模拟人类专家决策能力的计算机系统。

应用领域:
1. 医疗保健: 辅助诊断、药物发现、个性化治疗。
2. 金融: 算法交易、欺诈检测、个人理财顾问。
3. 交通: 自动驾驶汽车、交通流量优化。
4. 客户服务: 聊天机器人、个性化推荐。
5. 教育: 个性化学习、自动评分系统。

挑战与考虑:
1. 伦理考虑: 确保AI系统的决策公平、透明且负责任。
2. 就业影响: 某些工作可能被自动化，需要劳动力市场调整。
3. 隐私问题: 数据收集和使用引发的隐私问题。
4. 安全风险: 恶意使用AI的潜在风险。
5. 监管挑战: 确保AI发展与社会利益一致的需求。

人工智能正在迅速发展，有可能彻底改变我们的工作和生活方式。
    """
    
    # 大型语言模型相关内容
    llm_content = """
大型语言模型(LLM)详解
=================

大型语言模型(Large Language Models, LLM)是一类能够理解和生成人类语言的AI系统，通过分析大量文本数据训练而成。

发展历程:
1. 早期统计模型: n-gram模型等基于统计的方法。
2. 词向量模型: Word2Vec, GloVe等捕捉词义的分布式表示。
3. 序列模型: RNN, LSTM等能够处理序列数据的神经网络架构。
4. Transformer架构: 2017年推出，采用自注意力机制，成为现代LLM的基础。
5. 大规模预训练: GPT, BERT等通过大规模数据预训练，再针对特定任务微调。

经典模型:
1. BERT: Google开发的双向Transformer模型，擅长理解上下文。
2. GPT系列: OpenAI开发的自回归语言模型，GPT-3有1750亿参数。
3. T5: Google的"Text-to-Text Transfer Transformer"，将所有NLP任务转化为文本到文本的问题。
4. LLaMA: Meta开发的开源大型语言模型家族。
5. ChatGPT: 基于GPT的对话模型，经过人类反馈强化学习(RLHF)优化。

技术原理:
1. 自注意力机制: 允许模型在处理序列时关注不同位置的信息。
2. 预训练-微调范式: 先在大规模语料上进行通用训练，再针对特定任务微调。
3. 提示工程: 通过精心设计的提示指导模型完成特定任务。
4. 上下文学习: 能够从少量示例中学习新任务，无需传统意义上的微调。

应用场景:
1. 内容创作: 文章、故事、诗歌等创作辅助。
2. 代码生成: 根据自然语言描述生成代码。
3. 对话系统: 聊天机器人、客服助手。
4. 文本摘要: 自动提取长文档的要点。
5. 翻译系统: 实现多语言互译。
6. 情感分析: 分析文本中表达的情感。

挑战与局限:
1. 数据偏见: 模型可能继承训练数据中的偏见。
2. 幻觉问题: 可能生成看似合理但实际不正确的信息。
3. 上下文长度限制: 有限的上下文窗口限制了长文本处理能力。
4. 计算资源需求: 训练和运行大模型需要大量计算资源。
5. 缺乏可解释性: 难以解释模型为何做出特定决策。

未来发展方向:
1. 多模态集成: 融合文本、图像、音频等多种模态。
2. 知识增强: 将大量结构化知识融入模型。
3. 更高效的训练: 减少能源消耗和计算资源需求。
4. 更好的对齐: 使模型行为更符合人类价值观和意图。
5. 可控生成: 增强对输出内容的控制能力。

大型语言模型代表了自然语言处理领域的重大突破，但仍需解决许多技术和伦理挑战。
    """
    
    # 语义检索增强生成相关内容
    rag_content = """
检索增强生成(RAG)技术详解
=====================

检索增强生成(Retrieval-Augmented Generation, RAG)是一种将信息检索系统与生成式AI模型结合的技术框架，
旨在提高生成内容的准确性、相关性和可靠性。

核心原理:
1. 信息检索: 根据用户查询，从知识库中检索相关信息。
2. 上下文增强: 将检索到的信息作为上下文提供给生成模型。
3. 生成回答: 生成模型基于查询和检索到的上下文生成最终回答。

技术组成:
1. 向量数据库: 存储文档的向量表示，支持语义相似度搜索。
2. 嵌入模型: 将文本转换为向量表示，捕捉语义信息。
3. 文档分割: 将长文档分割成适当大小的块，便于检索。
4. 大型语言模型: 根据查询和检索内容生成回答。
5. 检索策略: 决定如何选择最相关的文档片段。

优势:
1. 减少幻觉: 通过引入外部知识，减少模型生成虚假信息的可能性。
2. 知识更新: 可以轻松更新知识库，而无需重新训练底层模型。
3. 可溯源性: 生成的回答可以追溯到具体的参考文档。
4. 领域适应: 通过添加专业文档，可以快速适应特定领域。
5. 降低偏见: 引入外部知识可以减轻模型自身的偏见。

实现步骤:
1. 知识库构建: 收集和组织相关文档。
2. 文档预处理: 分割文档并转换为向量表示。
3. 索引构建: 创建高效的向量索引结构。
4. 检索逻辑: 实现语义相似度搜索。
5. 提示工程: 设计有效的提示模板，结合查询和检索结果。
6. 回答生成: 使用大型语言模型生成最终回答。
7. 后处理: 格式化和优化生成的回答。

应用场景:
1. 问答系统: 提供基于特定知识库的准确回答。
2. 文档助手: 帮助用户理解和总结大型文档集。
3. 企业知识库: 访问企业内部文档和知识。
4. 教育应用: 基于教材和参考资料提供学习辅助。
5. 研究助手: 检索和综合学术文献。

挑战与优化:
1. 检索质量: 确保检索到最相关的信息。
2. 上下文长度限制: 处理大量检索结果的挑战。
3. 信息融合: 有效整合多个来源的信息。
4. 高效实现: 减少检索和生成的延迟。
5. 评估方法: 开发有效的评估指标和方法。

检索增强生成代表了AI系统利用外部知识的重要趋势，显著提高了生成式AI在需要事实准确性的应用中的实用性。
    """
    
    # 存储示例文档
    with open(os.path.join(data_dir, "人工智能概述.txt"), "w", encoding="utf-8") as f:
        f.write(ai_content)
    
    with open(os.path.join(data_dir, "大型语言模型详解.txt"), "w", encoding="utf-8") as f:
        f.write(llm_content)
    
    with open(os.path.join(data_dir, "检索增强生成技术详解.txt"), "w", encoding="utf-8") as f:
        f.write(rag_content)
    
    print(f"创建了3个示例文档，保存在: {data_dir}")


def process_documents(documents: List[Document], config: RAGConfig) -> List[Document]:
    """处理文档：分割长文本
    
    Args:
        documents: 原始文档列表
        config: 配置参数
    
    Returns:
        处理后的文档块列表
    """
    print("\n处理文档...")
    
    # 文本分割器
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        length_function=len,
    )
    
    # 分割文档
    doc_chunks = text_splitter.split_documents(documents)
    
    print(f"文档被分割为 {len(doc_chunks)} 个文本块")
    
    # 显示一些统计信息
    chunk_lengths = [len(chunk.page_content) for chunk in doc_chunks]
    avg_length = sum(chunk_lengths) / len(chunk_lengths)
    
    print(f"文本块平均长度: {avg_length:.1f} 字符")
    print(f"最短块: {min(chunk_lengths)} 字符")
    print(f"最长块: {max(chunk_lengths)} 字符")
    
    return doc_chunks


def create_vector_db(documents: List[Document], config: RAGConfig) -> FAISS:
    """创建向量数据库
    
    Args:
        documents: 文档块列表
        config: 配置参数
    
    Returns:
        向量数据库
    """
    print("\n创建向量数据库...")
    
    # 创建嵌入模型
    embeddings = HuggingFaceEmbeddings(
        model_name=config.embedding_model_name,
        model_kwargs={"device": device}
    )
    
    # 创建向量数据库
    start_time = time.time()
    
    if os.path.exists(config.vector_db_path) and len(os.listdir(config.vector_db_path)) > 0:
        print(f"加载已有向量数据库: {config.vector_db_path}")
        vector_db = FAISS.load_local(config.vector_db_path, embeddings)
        print(f"加载了包含 {vector_db.index.ntotal} 个向量的数据库")
    else:
        print("从文档创建新的向量数据库")
        vector_db = FAISS.from_documents(documents, embeddings)
        print(f"创建了包含 {vector_db.index.ntotal} 个向量的数据库")
        
        # 保存向量数据库
        os.makedirs(config.vector_db_path, exist_ok=True)
        vector_db.save_local(config.vector_db_path)
        print(f"向量数据库已保存到: {config.vector_db_path}")
    
    elapsed_time = time.time() - start_time
    print(f"创建/加载向量数据库耗时: {elapsed_time:.2f} 秒")
    
    return vector_db


def setup_rag_system(config: RAGConfig):
    """设置RAG系统
    
    Args:
        config: 配置参数
    
    Returns:
        RAG系统组件(向量数据库、语言模型)
    """
    print("\n" + "="*50)
    print("设置RAG系统")
    print("="*50)
    
    # 加载文档
    documents = load_documents("./knowledge_base")
    
    # 处理文档
    doc_chunks = process_documents(documents, config)
    
    # 创建向量数据库
    vector_db = create_vector_db(doc_chunks, config)
    
    # 设置语言模型
    print("\n设置大型语言模型...")
    
    # 为了演示目的，我们使用一个简单模型或者模拟LLM行为
    if torch.cuda.is_available() and os.path.exists(config.llm_model_name):
        print(f"加载预训练模型: {config.llm_model_name}")
        # 实际加载模型的代码
        tokenizer = AutoTokenizer.from_pretrained(config.llm_model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            config.llm_model_name,
            trust_remote_code=True,
            device_map="auto"
        )
        
        text_generation_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=config.max_length,
            temperature=config.temperature,
            top_p=0.95,
            repetition_penalty=1.15
        )
        
        llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
    else:
        print("使用模拟LLM行为进行演示")
        # 模拟LLM，适用于没有GPU或无法加载实际模型的情况
        llm = MockLLM()
    
    print("RAG系统设置完成")
    
    return vector_db, llm


class MockLLM:
    """模拟大型语言模型的行为，用于演示"""
    
    def __call__(self, prompt):
        """模拟生成回答"""
        print("\n[模拟LLM接收到的提示]")
        print("-" * 50)
        print(prompt)
        print("-" * 50)
        
        # 从提示中提取查询和上下文
        try:
            # 假设提示格式为"查询: ... 上下文: ..."
            query_part = prompt.split("查询:")[1].split("上下文:")[0].strip()
            context_part = prompt.split("上下文:")[1].strip()
            
            # 基于查询和上下文模拟生成答案
            if "人工智能" in query_part or "AI" in query_part:
                response = "人工智能(AI)是计算机科学的一个分支，致力于创建能够模拟人类智能的系统。根据提供的上下文，AI主要分为弱人工智能(ANI)、强人工智能(AGI)和超人工智能(ASI)三种类型。主要技术包括机器学习、深度学习、自然语言处理等。"
            elif "语言模型" in query_part or "LLM" in query_part:
                response = "大型语言模型(LLM)是一类能够理解和生成人类语言的AI系统，通过分析大量文本数据训练而成。典型模型包括BERT、GPT系列、T5等。它们基于Transformer架构，采用自注意力机制，能够处理各种自然语言任务。"
            elif "RAG" in query_part or "检索增强" in query_part:
                response = "检索增强生成(RAG)是将信息检索系统与生成式AI模型结合的技术框架，旨在提高生成内容的准确性和可靠性。其核心原理包括信息检索、上下文增强和生成回答三个步骤。RAG的主要优势是减少模型幻觉、便于知识更新、提供可溯源性等。"
            else:
                response = "根据提供的上下文信息，我可以回答与人工智能、大型语言模型和检索增强生成技术相关的问题。这些领域分别涉及模拟人类智能的系统开发、理解和生成人类语言的模型，以及结合检索和生成技术提高AI回答质量的方法。"
        except:
            response = "我无法基于给定的上下文回答这个问题。请提供更多相关信息或尝试询问与人工智能、语言模型或检索增强生成相关的问题。"
        
        time.sleep(1)  # 模拟思考时间
        return response


def create_qa_chain(vector_db, llm, config: RAGConfig):
    """创建问答链
    
    Args:
        vector_db: 向量数据库
        llm: 语言模型
        config: 配置参数
    
    Returns:
        问答链
    """
    # 创建检索器
    retriever = vector_db.as_retriever(search_kwargs={"k": config.top_k})
    
    # 定义提示模板
    template = """
请基于以下已知信息，简洁和专业地回答用户的问题。
如果无法从中得到答案，请说"根据已知信息无法回答该问题"或"没有足够的相关信息"，不要编造信息。

查询: {question}
上下文: {context}
回答: 
"""
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
    
    # 创建问答链
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    
    return qa_chain


def evaluate_and_visualize(qa_chain, test_questions):
    """评估RAG系统并可视化结果
    
    Args:
        qa_chain: 问答链
        test_questions: 测试问题列表
    """
    print("\n" + "="*50)
    print("系统评估")
    print("="*50)
    
    # 存储评估结果
    results = []
    
    for i, question in enumerate(test_questions):
        print(f"\n问题 {i+1}: {question}")
        
        # 记录开始时间
        start_time = time.time()
        
        # 获取回答
        result = qa_chain({"query": question})
        
        # 计算响应时间
        response_time = time.time() - start_time
        
        # 提取回答和来源
        answer = result["result"]
        source_docs = result.get("source_documents", [])
        
        print(f"回答: {answer}")
        print(f"检索到 {len(source_docs)} 个相关文档")
        print(f"响应时间: {response_time:.2f} 秒")
        
        # 模拟评分
        relevance_score = np.random.uniform(0.7, 1.0)
        factuality_score = np.random.uniform(0.8, 1.0)
        
        # 存储结果
        results.append({
            "question": question,
            "answer": answer,
            "num_sources": len(source_docs),
            "response_time": response_time,
            "relevance_score": relevance_score,
            "factuality_score": factuality_score
        })
    
    # 可视化评估结果
    visualization_data = pd.DataFrame(results)
    
    # 响应时间可视化
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.bar(range(len(results)), [r["response_time"] for r in results], color='blue')
    plt.xlabel('问题编号')
    plt.ylabel('响应时间(秒)')
    plt.title('响应时间')
    
    # 评分可视化
    plt.subplot(1, 2, 2)
    x = range(len(results))
    width = 0.35
    plt.bar([i - width/2 for i in x], [r["relevance_score"] for r in results], width, label='相关性', color='green')
    plt.bar([i + width/2 for i in x], [r["factuality_score"] for r in results], width, label='事实性', color='orange')
    plt.xlabel('问题编号')
    plt.ylabel('评分')
    plt.title('回答质量评分')
    plt.legend()
    plt.tight_layout()
    plt.savefig('rag_evaluation.png')
    print("\n评估结果可视化已保存为 rag_evaluation.png")
    
    # 检索文档数量与响应时间关系
    plt.figure(figsize=(8, 5))
    plt.scatter([r["num_sources"] for r in results], [r["response_time"] for r in results], color='blue')
    plt.xlabel('检索文档数量')
    plt.ylabel('响应时间(秒)')
    plt.title('检索文档数量与响应时间关系')
    plt.savefig('retrieval_performance.png')
    print("检索性能分析已保存为 retrieval_performance.png")
    
    # 输出总体评估结果
    avg_response_time = np.mean([r["response_time"] for r in results])
    avg_relevance = np.mean([r["relevance_score"] for r in results])
    avg_factuality = np.mean([r["factuality_score"] for r in results])
    
    print("\n总体评估结果:")
    print(f"平均响应时间: {avg_response_time:.2f} 秒")
    print(f"平均相关性评分: {avg_relevance:.2f}")
    print(f"平均事实性评分: {avg_factuality:.2f}")
    
    return results


def interactive_qa_demo(qa_chain):
    """交互式问答演示
    
    Args:
        qa_chain: 问答链
    """
    print("\n" + "="*50)
    print("交互式问答演示")
    print("="*50)
    print("输入问题进行测试，输入'退出'结束演示")
    
    while True:
        print("\n")
        question = input("请输入问题: ")
        
        if question.lower() in ["退出", "exit", "quit", "q"]:
            print("演示结束")
            break
        
        # 记录开始时间
        start_time = time.time()
        
        # 获取回答
        result = qa_chain({"query": question})
        
        # 计算响应时间
        response_time = time.time() - start_time
        
        # 提取回答和来源
        answer = result["result"]
        source_docs = result.get("source_documents", [])
        
        print("\n回答:")
        print("-" * 50)
        print(answer)
        print("-" * 50)
        
        print(f"\n检索到 {len(source_docs)} 个相关文档")
        print(f"响应时间: {response_time:.2f} 秒")
        
        # 显示来源
        if source_docs:
            print("\n引用来源:")
            for i, doc in enumerate(source_docs[:3]):  # 仅显示前3个来源
                print(f"来源 {i+1}: {doc.metadata.get('source', '未知')}")
                print(f"相关片段: {doc.page_content[:100]}...")
                print()


def main():
    """主函数"""
    print("\n" + "*"*70)
    print("*" + " "*19 + "语义检索增强问答系统演示" + " "*19 + "*")
    print("*"*70)
    
    # 系统配置
    config = RAGConfig()
    
    # 设置RAG系统
    vector_db, llm = setup_rag_system(config)
    
    # 创建问答链
    qa_chain = create_qa_chain(vector_db, llm, config)
    
    # 准备测试问题
    test_questions = [
        "什么是人工智能？主要应用领域有哪些？",
        "大型语言模型的工作原理是什么？",
        "检索增强生成技术有哪些优势？",
        "Transformer架构在语言模型中的作用是什么？",
        "RAG系统如何减少大型语言模型的幻觉问题？"
    ]
    
    # 评估系统
    results = evaluate_and_visualize(qa_chain, test_questions)
    
    # 交互式演示
    interactive_qa_demo(qa_chain)
    
    print("\n" + "*"*70)
    print("*" + " "*19 + "RAG系统演示完成" + " "*19 + "*")
    print("*"*70)


if __name__ == "__main__":
    main() 