import os
import getpass
import json
import base64
from cryptography.fernet import Fernet
from pathlib import Path

class APIKeyManager:
    """API密钥管理器类，用于加密存储和加载API密钥"""
    
    def __init__(self, config_file=None):
        """初始化API密钥管理器
        
        Args:
            config_file: 配置文件路径，默认为用户主目录下的.langchain_config.json
        """
        self.config_file = config_file or Path.home() / ".langchain_config.json"
    
    def encrypt_api_key(self, api_key, encryption_key):
        """加密API密钥"""
        f = Fernet(encryption_key)
        encrypted_key = f.encrypt(api_key.encode())
        return base64.b64encode(encrypted_key).decode()
    
    def decrypt_api_key(self, encrypted_key, encryption_key):
        """解密API密钥"""
        f = Fernet(encryption_key)
        decoded_key = base64.b64decode(encrypted_key)
        return f.decrypt(decoded_key).decode()
    
    def save_api_key(self, api_key):
        """保存加密的API密钥到配置文件"""
        # 生成加密密钥
        encryption_key = Fernet.generate_key()
        
        # 加密API密钥
        encrypted_key = self.encrypt_api_key(api_key, encryption_key)
        
        # 保存到配置文件
        config = {
            "encrypted_api_key": encrypted_key,
            "encryption_key": encryption_key.decode()
        }
        
        # 如果配置文件已存在，读取现有配置
        if self.config_file.exists():
            try:
                with open(self.config_file, "r") as f:
                    existing_config = json.load(f)
                config.update(existing_config)
            except Exception:
                pass
        
        with open(self.config_file, "w") as f:
            json.dump(config, f)
    
    def save_api_base(self, api_base):
        """保存加密的API基础URL到配置文件"""
        # 生成加密密钥
        encryption_key = Fernet.generate_key()
        
        # 加密API基础URL
        encrypted_base = self.encrypt_api_key(api_base, encryption_key)
        
        # 如果配置文件已存在，读取现有配置
        config = {}
        if self.config_file.exists():
            try:
                with open(self.config_file, "r") as f:
                    config = json.load(f)
            except Exception:
                pass
        
        # 更新配置
        config.update({
            "encrypted_api_base": encrypted_base,
            "api_base_encryption_key": encryption_key.decode()
        })
        
        with open(self.config_file, "w") as f:
            json.dump(config, f)
    
    def save_huggingface_api_key(self, api_key):
        """保存加密的HuggingFace API密钥到配置文件"""
        # 生成加密密钥
        encryption_key = Fernet.generate_key()
        
        # 加密API密钥
        encrypted_key = self.encrypt_api_key(api_key, encryption_key)
        
        # 如果配置文件已存在，读取现有配置
        config = {}
        if self.config_file.exists():
            try:
                with open(self.config_file, "r") as f:
                    config = json.load(f)
            except Exception:
                pass
        
        # 更新配置
        config.update({
            "encrypted_hf_api_key": encrypted_key,
            "hf_encryption_key": encryption_key.decode()
        })
        
        with open(self.config_file, "w") as f:
            json.dump(config, f)
    
    def save_custom_api_key(self, env_name, api_key):
        """保存加密的自定义API密钥到配置文件"""
        # 生成加密密钥
        encryption_key = Fernet.generate_key()
        
        # 加密API密钥
        encrypted_key = self.encrypt_api_key(api_key, encryption_key)
        
        # 如果配置文件已存在，读取现有配置
        config = {}
        if self.config_file.exists():
            try:
                with open(self.config_file, "r") as f:
                    config = json.load(f)
            except Exception:
                pass
        
        # 更新配置，使用环境变量名称作为键名前缀
        config.update({
            f"encrypted_{env_name}": encrypted_key,
            f"{env_name}_encryption_key": encryption_key.decode()
        })
        
        with open(self.config_file, "w") as f:
            json.dump(config, f)
    
    def load_api_key(self):
        """从配置文件加载API密钥"""
        if not self.config_file.exists():
            return None
        
        try:
            with open(self.config_file, "r") as f:
                config = json.load(f)
            
            if "encrypted_api_key" not in config or "encryption_key" not in config:
                return None
                
            encrypted_key = config["encrypted_api_key"]
            encryption_key = config["encryption_key"].encode()
            
            return self.decrypt_api_key(encrypted_key, encryption_key)
        except Exception as e:
            print(f"加载API密钥时出错: {e}")
            return None
    
    def load_api_base(self):
        """从配置文件加载API基础URL"""
        if not self.config_file.exists():
            return None
        
        try:
            with open(self.config_file, "r") as f:
                config = json.load(f)
            
            if "encrypted_api_base" not in config or "api_base_encryption_key" not in config:
                return None
                
            encrypted_base = config["encrypted_api_base"]
            encryption_key = config["api_base_encryption_key"].encode()
            
            return self.decrypt_api_key(encrypted_base, encryption_key)
        except Exception as e:
            print(f"加载API基础URL时出错: {e}")
            return None
    
    def load_huggingface_api_key(self):
        """从配置文件加载HuggingFace API密钥"""
        if not self.config_file.exists():
            return None
        
        try:
            with open(self.config_file, "r") as f:
                config = json.load(f)
            
            if "encrypted_hf_api_key" not in config or "hf_encryption_key" not in config:
                return None
                
            encrypted_key = config["encrypted_hf_api_key"]
            encryption_key = config["hf_encryption_key"].encode()
            
            return self.decrypt_api_key(encrypted_key, encryption_key)
        except Exception as e:
            print(f"加载HuggingFace API密钥时出错: {e}")
            return None
    
    def load_custom_api_key(self, env_name):
        """从配置文件加载自定义API密钥"""
        if not self.config_file.exists():
            return None
        
        try:
            with open(self.config_file, "r") as f:
                config = json.load(f)
            
            encrypted_key_name = f"encrypted_{env_name}"
            encryption_key_name = f"{env_name}_encryption_key"
            
            if encrypted_key_name not in config or encryption_key_name not in config:
                return None
                
            encrypted_key = config[encrypted_key_name]
            encryption_key = config[encryption_key_name].encode()
            
            return self.decrypt_api_key(encrypted_key, encryption_key)
        except Exception as e:
            print(f"加载{env_name}密钥时出错: {e}")
            return None
    
    def setup_api_key(self, env_names=None):
        """设置API密钥并配置环境变量
        
        Args:
            env_names: 要加载的环境变量名称，可以是None(加载所有)或列表(仅加载指定的环境变量)
        """
        # 确定需要加载的环境变量
        load_all = env_names is None
        env_list = env_names if isinstance(env_names, list) else []
        
        # 设置OpenAI API密钥
        if load_all or "OPENAI_API_KEY" in env_list:
            # 尝试从配置文件加载API密钥
            api_key = self.load_api_key()
            
            # 如果没有保存的密钥，则请求用户输入
            if not api_key:
                api_key = getpass.getpass("请输入您的OpenAI API密钥: ")
                save_choice = input("是否保存密钥以便下次使用? (y/n): ").lower()
                if save_choice == 'y':
                    self.save_api_key(api_key)
                    print("API密钥已加密保存")
            
            # 设置环境变量
            os.environ["OPENAI_API_KEY"] = api_key
        
        # 设置API基础URL
        if load_all or "OPENAI_API_BASE" in env_list:
            api_base = self.load_api_base()
            if not api_base:
                api_base = input("请输入API基础URL (默认为https://api.deepseek.com/v1): ") or "https://api.deepseek.com/v1"
                save_choice = input("是否保存API基础URL以便下次使用? (y/n): ").lower()
                if save_choice == 'y':
                    self.save_api_base(api_base)
                    print("API基础URL已加密保存")
            
            os.environ["OPENAI_API_BASE"] = api_base     # DeepSeek API BASE
        
        # 设置HuggingFace API密钥
        if load_all or "HUGGINGFACEHUB_API_TOKEN" in env_list:
            hf_api_key = self.load_huggingface_api_key()
            if not hf_api_key:
                hf_api_key = getpass.getpass("请输入您的HuggingFace API密钥 (如果没有请直接回车): ")
                if hf_api_key:
                    save_choice = input("是否保存HuggingFace密钥以便下次使用? (y/n): ").lower()
                    if save_choice == 'y':
                        self.save_huggingface_api_key(hf_api_key)
                        print("HuggingFace API密钥已加密保存")
            
            if hf_api_key:
                os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_api_key
        
        # 处理其他自定义环境变量
        known_env_vars = ["OPENAI_API_KEY", "OPENAI_API_BASE", "HUGGINGFACEHUB_API_TOKEN"]
        custom_env_vars = [var for var in env_list if var not in known_env_vars] if env_list else []
        
        for env_var in custom_env_vars:
            # 尝试从配置文件加载自定义API密钥
            api_key = self.load_custom_api_key(env_var)
            
            # 如果没有保存的密钥，则请求用户输入
            if not api_key:
                api_key = getpass.getpass(f"请输入您的{env_var}密钥: ")
                if api_key:
                    save_choice = input(f"是否保存{env_var}密钥以便下次使用? (y/n): ").lower()
                    if save_choice == 'y':
                        self.save_custom_api_key(env_var, api_key)
                        print(f"{env_var}密钥已加密保存")
            
            # 设置环境变量
            if api_key:
                os.environ[env_var] = api_key
        
        return api_key


# 使用示例
if __name__ == "__main__":
    key_manager = APIKeyManager()
    key_manager.setup_api_key()