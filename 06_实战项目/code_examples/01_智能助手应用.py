#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
智能助手应用示例
=================
本代码演示了基于LLM的智能助手应用的构建，包括：
1. 应用架构设计
2. 对话管理与状态维护
3. 工具集成与调用
4. 多模态信息处理
5. 简单Web界面实现
"""

import os
import sys
import time
import json
import logging
import asyncio
import requests
from typing import List, Dict, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
import argparse
import re
import random
from pathlib import Path

# Web界面相关依赖
import gradio as gr

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("assistant.log")
    ]
)
logger = logging.getLogger("LLM-Assistant")


@dataclass
class AssistantConfig:
    """助手配置参数"""
    # 模型配置
    model_name: str = "gpt-3.5-turbo"  # 实际使用时替换为实际可用的模型
    api_key: str = None  # 实际使用时需要设置
    api_base: str = "https://api.openai.com/v1"
    temperature: float = 0.7
    max_tokens: int = 1000
    
    # 助手配置
    assistant_name: str = "智能助手"
    system_prompt: str = "你是一个有用、安全、诚实的AI助手。回答用户问题时，请提供准确、有帮助的信息。"
    
    # 工具配置
    available_tools: List[str] = field(default_factory=lambda: [
        "search", "calculator", "weather", "calendar", "translator"
    ])
    
    # 记忆配置
    memory_size: int = 10  # 保存的对话轮数
    
    # 界面配置
    theme: str = "default"
    language: str = "zh-CN"
    
    # 数据存储
    data_dir: str = "./assistant_data"
    conversation_file: str = "conversations.json"
    
    def __post_init__(self):
        """初始化后处理"""
        # 从环境变量加载API密钥（如果存在）
        if not self.api_key:
            self.api_key = os.environ.get("OPENAI_API_KEY", "")


class MockLLM:
    """模拟LLM，用于演示和测试"""
    
    def __init__(self, config: AssistantConfig):
        """初始化模型"""
        self.config = config
        logger.info(f"初始化模拟LLM: {config.model_name}")
        
        # 模拟响应模板
        self.templates = {
            "greeting": [
                "你好！我是{assistant_name}，有什么我可以帮助你的吗？",
                "你好！我能为你做些什么？",
                "你好，今天我能帮你解决什么问题？"
            ],
            "farewell": [
                "再见！如果有其他问题，随时回来。",
                "祝你有美好的一天！",
                "再见！很高兴能帮到你。"
            ],
            "search": [
                "我找到关于'{query}'的以下信息：\n\n{result}",
                "根据我的搜索，关于'{query}'的信息如下：\n\n{result}",
                "以下是关于'{query}'的搜索结果：\n\n{result}"
            ],
            "weather": [
                "{location}今天的天气是{weather}，温度{temperature}°C。",
                "今天{location}的天气预报：{weather}，温度约{temperature}°C。",
                "{location}的当前天气状况：{weather}，温度{temperature}°C。"
            ],
            "calculator": [
                "计算结果是: {result}",
                "{expression} = {result}",
                "经过计算，{expression}的结果是{result}。"
            ],
            "unknown": [
                "我理解你的问题，但需要更多信息才能提供准确回答。能否详细说明？",
                "我不太确定你的问题，能否重新表述一下？",
                "抱歉，我无法完全理解。能否提供更多上下文？"
            ]
        }
    
    def _generate_mock_response(self, user_input: str, conversation_history: List[Dict]) -> str:
        """生成模拟回复"""
        # 检查是否是问候语
        if re.search(r'你好|嗨|hi|hello', user_input.lower()):
            template = random.choice(self.templates["greeting"])
            return template.format(assistant_name=self.config.assistant_name)
        
        # 检查是否是告别语
        if re.search(r'再见|拜拜|goodbye|bye', user_input.lower()):
            return random.choice(self.templates["farewell"])
        
        # 检查是否是搜索请求
        search_match = re.search(r'搜索|查询|查找|了解|什么是|谁是|(.+的信息)', user_input)
        if search_match:
            query = search_match.group(1) if search_match.group(1) else user_input
            mock_result = f"- {query}是一个重要概念\n- 它有广泛的应用\n- 近年来研究有所突破"
            template = random.choice(self.templates["search"])
            return template.format(query=query, result=mock_result)
        
        # 检查是否是天气请求
        weather_match = re.search(r'(天气|气温|温度).*(北京|上海|广州|深圳|杭州)', user_input)
        if weather_match:
            location = weather_match.group(2)
            weather_conditions = ["晴朗", "多云", "小雨", "阴天"]
            template = random.choice(self.templates["weather"])
            return template.format(
                location=location, 
                weather=random.choice(weather_conditions), 
                temperature=random.randint(15, 30)
            )
        
        # 检查是否是计算请求
        calc_match = re.search(r'计算|等于|是多少|([0-9]+[\+\-\*/][0-9]+)', user_input)
        if calc_match:
            expression = calc_match.group(1) if calc_match.group(1) else "1+1"
            try:
                result = eval(expression)
                template = random.choice(self.templates["calculator"])
                return template.format(expression=expression, result=result)
            except:
                pass
        
        # 默认回复
        return random.choice(self.templates["unknown"])
    
    async def generate(self, user_input: str, conversation_history: List[Dict]) -> str:
        """异步生成回复"""
        # 模拟网络延迟
        await asyncio.sleep(1)
        return self._generate_mock_response(user_input, conversation_history)


class AssistantMemory:
    """对话历史管理器"""
    
    def __init__(self, config: AssistantConfig):
        """初始化记忆管理器"""
        self.config = config
        self.memory = []
        self.file_path = os.path.join(config.data_dir, config.conversation_file)
        
        # 确保数据目录存在
        os.makedirs(config.data_dir, exist_ok=True)
        
        # 尝试加载已有对话
        self.load_conversations()
    
    def add(self, role: str, content: str) -> None:
        """添加新的对话"""
        self.memory.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        
        # 限制记忆大小
        if len(self.memory) > self.config.memory_size * 2:  # 每轮对话有用户和助手两条消息
            self.memory = self.memory[-self.config.memory_size * 2:]
    
    def get_conversation_history(self) -> List[Dict]:
        """获取对话历史"""
        return self.memory
    
    def clear(self) -> None:
        """清空对话历史"""
        self.memory = []
    
    def save_conversations(self) -> None:
        """保存对话历史到文件"""
        try:
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump(self.memory, f, ensure_ascii=False, indent=2)
            logger.info(f"对话已保存到: {self.file_path}")
        except Exception as e:
            logger.error(f"保存对话失败: {str(e)}")
    
    def load_conversations(self) -> None:
        """从文件加载对话历史"""
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    self.memory = json.load(f)
                logger.info(f"从 {self.file_path} 加载了 {len(self.memory)} 条对话")
            except Exception as e:
                logger.error(f"加载对话失败: {str(e)}")


class ToolManager:
    """工具管理器，负责识别和调用外部工具"""
    
    def __init__(self, config: AssistantConfig):
        """初始化工具管理器"""
        self.config = config
        
        # 工具映射表
        self.tools = {
            "search": self.search_tool,
            "calculator": self.calculator_tool,
            "weather": self.weather_tool,
            "calendar": self.calendar_tool,
            "translator": self.translator_tool
        }
        
        logger.info(f"工具管理器初始化完成，可用工具: {', '.join(config.available_tools)}")
    
    def detect_tool_calls(self, text: str) -> List[Dict]:
        """检测文本中的工具调用请求"""
        tool_calls = []
        
        # 检测搜索调用
        if re.search(r'搜索|查找|查询', text):
            query = re.search(r'搜索\s+(.+)', text)
            if query:
                tool_calls.append({
                    "tool": "search",
                    "params": {"query": query.group(1)}
                })
        
        # 检测计算器调用
        calc_match = re.search(r'计算\s+(.+)', text)
        if calc_match:
            tool_calls.append({
                "tool": "calculator",
                "params": {"expression": calc_match.group(1)}
            })
        
        # 检测天气调用
        weather_match = re.search(r'(查询|获取|如何|怎么样|今天|明天).*?天气.*?(?:在|at)?\s*(.+)', text)
        if weather_match:
            tool_calls.append({
                "tool": "weather",
                "params": {"location": weather_match.group(2)}
            })
        
        return tool_calls
    
    async def execute_tool(self, tool_name: str, params: Dict) -> Dict:
        """执行工具调用"""
        if tool_name not in self.config.available_tools:
            return {"error": f"工具 {tool_name} 不可用"}
        
        if tool_name not in self.tools:
            return {"error": f"工具 {tool_name} 未实现"}
        
        try:
            result = await self.tools[tool_name](**params)
            return {"result": result}
        except Exception as e:
            logger.error(f"工具 {tool_name} 执行失败: {str(e)}")
            return {"error": f"工具执行失败: {str(e)}"}
    
    async def search_tool(self, query: str) -> str:
        """搜索工具（模拟）"""
        await asyncio.sleep(1)  # 模拟网络延迟
        return f"关于 '{query}' 的搜索结果：\n1. {query} 的基本介绍\n2. {query} 的历史发展\n3. {query} 的应用案例"
    
    async def calculator_tool(self, expression: str) -> str:
        """计算器工具"""
        # 安全检查表达式
        if not re.match(r'^[\d\+\-\*/\(\)\s\.]+$', expression):
            return "表达式包含不安全字符"
        
        try:
            result = eval(expression)
            return f"{expression} = {result}"
        except Exception as e:
            return f"计算错误: {str(e)}"
    
    async def weather_tool(self, location: str) -> str:
        """天气工具（模拟）"""
        await asyncio.sleep(1)  # 模拟网络延迟
        weather_conditions = ["晴朗", "多云", "小雨", "阴天", "大风"]
        temperature = random.randint(15, 30)
        
        return f"{location}的天气：{random.choice(weather_conditions)}，温度{temperature}°C"
    
    async def calendar_tool(self, date: str = None) -> str:
        """日历工具"""
        if not date:
            date = datetime.now().strftime("%Y-%m-%d")
        
        try:
            dt = datetime.strptime(date, "%Y-%m-%d")
            return f"{date}是{dt.strftime('%A')}，{dt.year}年的第{dt.strftime('%j')}天"
        except Exception as e:
            return f"日期解析错误: {str(e)}"
    
    async def translator_tool(self, text: str, target_language: str = "英语") -> str:
        """翻译工具（模拟）"""
        await asyncio.sleep(1)  # 模拟网络延迟
        languages = {
            "英语": {"你好": "Hello", "谢谢": "Thank you", "再见": "Goodbye"},
            "法语": {"你好": "Bonjour", "谢谢": "Merci", "再见": "Au revoir"},
            "日语": {"你好": "こんにちは", "谢谢": "ありがとう", "再见": "さようなら"},
        }
        
        if target_language not in languages:
            return f"不支持的目标语言: {target_language}"
        
        if text in languages[target_language]:
            return languages[target_language][text]
        
        return f"[模拟]将'{text}'翻译为{target_language}"


class Assistant:
    """智能助手主类"""
    
    def __init__(self, config: AssistantConfig = None):
        """初始化助手"""
        if not config:
            config = AssistantConfig()
        
        self.config = config
        self.llm = MockLLM(config)  # 实际应用中替换为真实LLM
        self.memory = AssistantMemory(config)
        self.tools = ToolManager(config)
        
        logger.info(f"智能助手 '{config.assistant_name}' 初始化完成")
    
    async def process_message(self, user_input: str) -> str:
        """处理用户消息"""
        logger.info(f"收到用户消息: {user_input}")
        
        # 将用户消息添加到记忆
        self.memory.add("user", user_input)
        
        # 检测是否需要调用工具
        tool_calls = self.tools.detect_tool_calls(user_input)
        tool_results = ""
        
        if tool_calls:
            logger.info(f"检测到工具调用: {tool_calls}")
            
            # 执行每个工具调用
            for call in tool_calls:
                result = await self.tools.execute_tool(call["tool"], call["params"])
                
                if "result" in result:
                    tool_results += f"\n\n工具 '{call['tool']}' 的结果:\n{result['result']}"
                else:
                    tool_results += f"\n\n工具 '{call['tool']}' 调用失败: {result['error']}"
        
        # 获取对话历史
        conversation_history = self.memory.get_conversation_history()
        
        # 生成回复
        model_response = await self.llm.generate(user_input, conversation_history)
        
        # 如果有工具结果，添加到回复中
        if tool_results:
            response = model_response + tool_results
        else:
            response = model_response
        
        # 将助手回复添加到记忆
        self.memory.add("assistant", response)
        
        # 保存对话
        self.memory.save_conversations()
        
        return response
    
    def create_web_interface(self) -> gr.Blocks:
        """创建Web界面"""
        with gr.Blocks(title=self.config.assistant_name) as interface:
            gr.Markdown(f"# {self.config.assistant_name}")
            
            chatbot = gr.Chatbot(height=400)
            msg = gr.Textbox(placeholder="输入你的问题...")
            clear = gr.Button("清空对话")
            
            async def respond(message, chat_history):
                if not message:
                    return "", chat_history
                
                bot_response = await self.process_message(message)
                chat_history.append((message, bot_response))
                return "", chat_history
            
            def clear_history():
                self.memory.clear()
                return []
            
            msg.submit(respond, [msg, chatbot], [msg, chatbot])
            clear.click(clear_history, None, chatbot)
            
            # 加载已有对话
            conversation = self.memory.get_conversation_history()
            if conversation:
                chat_history = []
                for i in range(0, len(conversation), 2):
                    if i+1 < len(conversation):
                        chat_history.append((
                            conversation[i]["content"], 
                            conversation[i+1]["content"]
                        ))
                
                chatbot.value = chat_history
            
        return interface
    
    def launch_web_interface(self, server_port: int = 7860):
        """启动Web界面"""
        interface = self.create_web_interface()
        interface.launch(server_port=server_port)
        
    def run_cli(self):
        """运行命令行界面"""
        print(f"\n=== {self.config.assistant_name} ===")
        print("输入'exit'或'quit'退出程序\n")
        
        async def cli_loop():
            while True:
                try:
                    user_input = input("\n你: ")
                    
                    if user_input.lower() in ["exit", "quit", "退出"]:
                        print(f"{self.config.assistant_name}: 再见！")
                        break
                    
                    response = await self.process_message(user_input)
                    print(f"\n{self.config.assistant_name}: {response}")
                    
                except KeyboardInterrupt:
                    print("\n程序被中断")
                    break
                except Exception as e:
                    print(f"\n发生错误: {str(e)}")
        
        asyncio.run(cli_loop())


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="智能助手应用")
    parser.add_argument("--web", action="store_true", help="启动Web界面")
    parser.add_argument("--port", type=int, default=7860, help="Web服务端口")
    args = parser.parse_args()
    
    # 创建助手配置
    config = AssistantConfig()
    
    # 初始化助手
    assistant = Assistant(config)
    
    if args.web:
        # 启动Web界面
        assistant.launch_web_interface(server_port=args.port)
    else:
        # 启动命令行界面
        assistant.run_cli()


if __name__ == "__main__":
    main() 