import requests
import os
import random
import hashlib  
import json  
from server import PromptServer
from aiohttp import web 
import folder_paths  
import numpy as np  
import torch  
import re
import base64
import io
from PIL import Image, ImageOps


class OllamaPromptsGeneratorTlant:  
    """  
    A node to concatenate three input strings.  

    Class methods  
    -------------  
    INPUT_TYPES (dict):  
        Defines the input parameters of the node.  

    Attributes  
    ----------  
    RETURN_TYPES (`tuple`):  
        The type of each element in the output tuple.  
    FUNCTION (`str`):  
        The name of the entry-point method.  
    CATEGORY (`str`):  
        The category the node should appear in the UI.  

    Methods  
    -------  
    execute(s) -> tuple:  
        Concatenates three strings and returns the result.  
    """  
    def __init__(self):
        pass

    @classmethod  
    def INPUT_TYPES(cls):  
        """  
        Defines input fields and their configurations.  
        """  
        return {  
            "required": {  
                "positive_text": ("STRING", {"forceInput": True}),
                "reference_text": ("STRING", {"forceInput": True}),
                "negative_text": ("STRING", {"forceInput": True}),
                "prompt_template": (
                    "STRING", 
                    {
                        "forceInput": True,
                        "default": 
                        """As a professional art critic, describe the image based on the descriptions provided in the following three reference paragraphs. 
The first reference paragraph:
{positive_text}
The second reference paragraph:
{reference_text}
The third reference paragraph:
{negative_text}
Create a coherent and realistic scene in a single paragraph. Include:
Describe placement
Main subject details
Artistic style and theme
Setting and narrative contribution
Lighting characteristics
Color palette and emotional tone
Camera angle and focus
The elements described in the first reference paragraph must be included in the output paragraph, and if there is a conflict between the first reference paragraph and the second reference paragraph, the first reference paragraph takes precedence.
The description in the third reference paragraph contains negative elements and should not appear in the output paragraph. If no description for the third reference paragraph is provided, it can be ignored.
Merge image concepts if there is more than one.
Always blend the concepts, never talk about splits or parallel. 
Do not split or divide scenes, or talk about them differently - merge everything to one scene and one scene only.
Blend all elements into unified reality. Use image generation prompt language. No preamble, questions, or commentary."""
                    }
                ),
            },  
            "optional": {
                "custom_model": ("STRING", {"default": "llama3.1:8b"}),
                "ollama_url": (
                    "STRING",
                    {"default": "http://localhost:11434/api/generate"},
                ),
            },
        }  

    RETURN_TYPES = ("STRING",)  
    FUNCTION = "execute"  
    CATEGORY = "prompts"  

    @classmethod  
    def IS_CHANGED(cls, directory_path):  
        """  
        每次执行时强制节点重新运行。  
        返回一个随机数作为变化标识。  
        """  
        return str(random.random())  

    def execute(self, positive_text, reference_text, negative_text, prompt_template, custom_model, ollama_url):  

        prompt = prompt_template.format(positive_text=positive_text, reference_text=reference_text, negative_text=negative_text)

        payload = {"model": custom_model, "prompt": prompt, "stream": False}

        response = requests.post(ollama_url, json=payload)
        response.raise_for_status()
        result = response.json()["response"]
        return (result,)  


class LoadRandomTxtFileTlant:
    """  
    随机读取指定目录下的一个txt文件的内容并返回。  

    类方法  
    -------  
    INPUT_TYPES (dict):  
        定义节点的输入参数。  
    IS_CHANGED:  
        强制节点每次执行时重新运行。  

    属性  
    -------  
    RETURN_TYPES (`tuple`):  
        输出的内容类型，这里为字符串类型。  
    RETURN_NAMES (`tuple`):  
        输出内容的名称。  
    FUNCTION (`str`):  
        节点的入口方法名称。  
    CATEGORY (`str`):  
        节点在UI中的分类。  
    """  

    def __init__(self):  
        pass  

    @classmethod  
    def INPUT_TYPES(cls):  
        """  
        定义输入参数为字符串类型的文件路径。  
        """  
        return {  
            "required": {  
                "directory_path": ("STRING", {  
                    "multiline": False,  
                    "default": "",  
                    "lazy": True  
                }),  
            },  
        }  

    RETURN_TYPES = ("STRING", )  
    RETURN_NAMES = ("file_content", )  
    FUNCTION = "execute"  
    CATEGORY = "Custom Nodes"  

    @classmethod  
    def IS_CHANGED(cls, directory_path):  
        """  
        每次执行时强制节点重新运行，确保随机选择文件。  
        返回一个随机数作为变化标识。  
        """  
        return str(random.random())  

    def execute(self, directory_path):  
        """  
        执行方法：  
        1. 检查路径是否存在且为目录。  
        2. 列出目录下所有txt文件。  
        3. 随机选择一个文件并读取内容。  
        4. 返回文件内容。  
        """  
        if not os.path.isdir(directory_path):  
            return ("指定的路径不是一个有效的目录。", )  

        txt_files = [f for f in os.listdir(directory_path) if f.endswith('.txt')]  

        if not txt_files:  
            return ("指定目录下没有txt文件。", )  

        selected_file = random.choice(txt_files)  
        file_path = os.path.join(directory_path, selected_file)  

        try:  
            with open(file_path, 'r', encoding='utf-8') as file:  
                content = file.read()  
            return (content, )  
        except Exception as e:  
            return (f"读取文件时发生错误: {e}", )  
     
   
class OllamaSimpleTextGeneratorTlant:  
    def __init__(self):  
        pass  

    @classmethod  
    def INPUT_TYPES(cls):  
        return {  
            "required": {  
                "text1": ("STRING", {"forceInput": True}),  
                "text2": ("STRING", {"forceInput": True}),  
                "prompt_template": ("STRING", {  
                    "multiline": True,  
                    "default": "By analyzing the content of Text 1 and Text 2, combine them into a coherent description.Text 1: {text1} Text 2: {text2}\n Give me result only."  
                }),  
            },  
            "optional": {  
                "model": ("STRING", {"default": "llama3"}),  
                "api_url": ("STRING", {"default": "http://localhost:11434/api/generate"}),  
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.1}),  
            }  
        }  

    RETURN_TYPES = ("STRING",)  
    RETURN_NAMES = ("generated_text",)  
    FUNCTION = "generate"  
    CATEGORY = "AI Tools/Ollama"  

    @classmethod  
    def IS_CHANGED(cls, *args, **kwargs):  
        # 生成参数指纹  
        param_str = str(kwargs) + str(args)  
        return hashlib.sha256(param_str.encode()).hexdigest()  

    def generate(self, text1, text2, prompt_template, model, api_url, temperature):  
        # 模板格式化  
        final_prompt = prompt_template.format(  
            text1=text1.strip(),  
            text2=text2.strip(),  
            model=model,  
            url=api_url  
        )  
        
        # API请求配置  
        payload = {  
            "model": model,  
            "prompt": final_prompt,  
            "stream": False,  
            "options": {"temperature": temperature}  
        }  

        try:  
            response = requests.post(api_url,   
                json=payload,  
                headers={"Content-Type": "application/json"},  
                timeout=30  
            )  
            response.raise_for_status()  
            return (response.json()["response"].strip(),)  
        except requests.exceptions.RequestException as e:  
            return (f"Error: {str(e)}",)  
        
        
class LoadRandomTxtFileTlantV2:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dir_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Enter directory path"
                }),
                "seed": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "step": 1,
                    "display": "number"
                }),
                "is_recursive": (["true", "false"], {
                    "default": "false"
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("text", "txt_file", "seed")
    FUNCTION = "read_file"
    CATEGORY = "Custom Nodes/File Operations"

    def read_file(self, dir_path, seed, is_recursive):
        txt_files = []
        is_recursive = (is_recursive == "true")

        try:
            # 路径有效性验证
            if not os.path.exists(dir_path):
                return (f"Directory not exists: {dir_path}", "", seed)
                
            if not os.path.isdir(dir_path):
                return (f"Not a directory: {dir_path}", "", seed)

            # 文件收集逻辑
            if is_recursive:
                for root, _, files in os.walk(dir_path):
                    for file in files:
                        if file.lower().endswith('.txt'):
                            full_path = os.path.abspath(os.path.join(root, file))
                            if os.path.isfile(full_path):
                                txt_files.append(full_path)
            else:
                for file in os.listdir(dir_path):
                    if file.lower().endswith('.txt'):
                        full_path = os.path.abspath(os.path.join(dir_path, file))
                        if os.path.isfile(full_path):
                            txt_files.append(full_path)

            if not txt_files:
                return ("No .txt files found in directory", "", seed)

        except PermissionError as e:
            return (f"Permission denied: {str(e)}", "", seed)
        except Exception as e:
            return (f"System error: {str(e)}", "", seed)

        # 随机选择文件
        random.seed(seed)
        selected_file = random.choice(txt_files)
        abs_path = os.path.abspath(selected_file)  # 确保绝对路径

        # 文件读取逻辑
        try:
            for encoding in ['utf-8', 'gbk', 'latin-1']:
                try:
                    with open(selected_file, 'r', encoding=encoding) as f:
                        content = f.read()
                        return (content, abs_path, seed)
                except UnicodeDecodeError:
                    continue
            return ("Failed to decode text file", abs_path, seed)
        except Exception as e:
            return (f"File read error: {str(e)}", abs_path, seed)
        
class LoadRandomTxtFileTlantV3:  
    def __init__(self):  
        self.sub_directories = []  
    
    @classmethod  
    def INPUT_TYPES(cls):  
        return {  
            "required": {  
                "dir_path": ("STRING", {  
                    "default": "",  
                    "multiline": False,  
                    "placeholder": "Enter directory path"  
                }),  
                "seed": ("INT", {  
                    "default": 1,  
                    "min": 0,  
                    "max": 0xffffffffffffffff,  
                    "step": 1,  
                    "display": "number"  
                }),  
                "is_recursive": (["true", "false"], {  
                    "default": "false"  
                }),  
                # 新增参数 sub_dir_list  
                "sub_dir_list": ("STRING", {  
                    "default": "",  
                    "multiline": False,  
                }),   
                # 新增参数 selected_sub_dir  
                "selected_sub_dir": ("STRING", {  
                    "default": "",  
                    "multiline": False,  
                    "placeholder": "Selected directory path"  
                }),  
            },  
        }  
    

    RETURN_TYPES = ("STRING", "STRING", "INT")  
    RETURN_NAMES = ("text", "txt_file", "seed")  
    FUNCTION = "read_file"  
    CATEGORY = "Custom Nodes/File Operations"  

    def read_file(self, dir_path, seed, is_recursive, sub_dir_list="", selected_sub_dir=""):  
        # 使用 selected_sub_dir 作为实际路径  
        working_dir = selected_sub_dir if selected_sub_dir else dir_path  
        
        txt_files = []  
        is_recursive = (is_recursive == "true")  

        try:  
            # 路径有效性验证  
            if not os.path.exists(working_dir):  
                return (f"Directory not exists: {working_dir}", "", seed)  
                
            if not os.path.isdir(working_dir):  
                return (f"Not a directory: {working_dir}", "", seed)  

            # 文件收集逻辑  
            if is_recursive:  
                for root, _, files in os.walk(working_dir):  
                    for file in files:  
                        if file.lower().endswith('.txt'):  
                            full_path = os.path.abspath(os.path.join(root, file))  
                            if os.path.isfile(full_path):  
                                txt_files.append(full_path)  
            else:  
                for file in os.listdir(working_dir):  
                    if file.lower().endswith('.txt'):  
                        full_path = os.path.abspath(os.path.join(working_dir, file))  
                        if os.path.isfile(full_path):  
                            txt_files.append(full_path)  

            if not txt_files:  
                return ("No .txt files found in directory", "", seed)  

        except PermissionError as e:  
            return (f"Permission denied: {str(e)}", "", seed)  
        except Exception as e:  
            return (f"System error: {str(e)}", "", seed)  

        # 随机选择文件  
        random.seed(seed)  
        selected_file = random.choice(txt_files)  
        abs_path = os.path.abspath(selected_file)  # 确保绝对路径  

        # 文件读取逻辑  
        try:  
            for encoding in ['utf-8', 'gbk', 'latin-1']:  
                try:  
                    with open(selected_file, 'r', encoding=encoding) as f:  
                        content = f.read()  
                        return (content, abs_path, seed)  
                except UnicodeDecodeError:  
                    continue  
            return ("Failed to decode text file", abs_path, seed)  
        except Exception as e:  
            return (f"File read error: {str(e)}", abs_path, seed)  

# 注册 API 路由来处理文件夹加载请求  
@PromptServer.instance.routes.post("/get_subdirectories")  
async def get_subdirectories(request):  
    data = await request.json()  
    base_dir = data.get("dir_path", "")  
    
    if not base_dir or not os.path.exists(base_dir) or not os.path.isdir(base_dir):  
        # 使用web.json_response返回JSON数据  
        return web.json_response({"subdirs": [""]})  
    
    subdirs = [""]  # 包含空选项，代表根目录自身  
    
    try:  
        # 遍历所有子目录  
        for root, dirs, _ in os.walk(base_dir):  
            for dir_name in dirs:  
                full_path = os.path.join(root, dir_name)  
                # 计算相对路径  
                rel_path = os.path.relpath(full_path, base_dir)  
                # 使用标准的路径分隔符  
                std_rel_path = rel_path.replace("\\", "/")  
                subdirs.append(std_rel_path)  
    except Exception as e:  
        print(f"Error getting subdirectories: {str(e)}")  
    
    # 使用web.json_response返回JSON数据  
    return web.json_response({"subdirs": sorted(subdirs)})  
    return {"subdirs": sorted(subdirs)}  


class LoadImageAndExtractMetadataTlant:  
    @classmethod  
    def INPUT_TYPES(cls):  
        return {  
            "required": {  
                # 删除默认为None的设置，因为ComfyUI可能不接受这种形式  
            },  
            "optional": {  
                "image_path": ("STRING", {"default": ""})  
            }  
        }  
    
    CATEGORY = "image"  
    FUNCTION = "load_image_and_extract_metadata"  
    RETURN_TYPES = ("IMAGE", "STRING", "JSON")  # JSON类型需要ComfyUI支持，若不支持请改为STRING  
    RETURN_NAMES = ("image", "image_path", "metadata_json")  
    
    # 这个方法允许节点显示上传按钮  
    @classmethod  
    def IS_CHANGED(cls, image_path):  
        if image_path != "":  
            m = hashlib.md5()  
            with open(image_path, 'rb') as f:  
                m.update(f.read())  
            return m.digest().hex()  
        return ""  
    
    # 这个方法处理上传的图片  
    @classmethod  
    def UPLOAD_HANDLER(cls, file):  
        filename = os.path.basename(file.orig_name)  
        
        # 保存上传的文件到ComfyUI的临时目录  
        temp_dir = folder_paths.get_temp_directory()  
        file_path = os.path.join(temp_dir, filename)  
        
        with open(file_path, "wb") as f:  
            f.write(file.file.read())  
        
        return {"image_path": file_path}  
    
    def load_image_and_extract_metadata(self, image_path=""):  
        # 如果提供了图像路径，从路径加载图像  
        if image_path == "":  
            raise ValueError("No image path provided")  
            
        if not os.path.exists(image_path):  
            raise FileNotFoundError(f"Image not found: {image_path}")  
        
        i = Image.open(image_path)  
        i = ImageOps.exif_transpose(i)  
        image = i.convert("RGB")  
        image = np.array(image).astype(np.float32) / 255.0  
        image = torch.from_numpy(image)[None,]  
        
        # 提取元数据  
        metadata_json = self.extract_metadata_from_image(image_path)  
        
        return (image, image_path, metadata_json)  
    
    def extract_metadata_from_image(self, image_path):  
        if image_path is None:  
            return "{}"  
        
        try:  
            # 打开PNG图像  
            img = Image.open(image_path)  
            
            # 尝试获取PNG文本块中的元数据  
            metadata = None  
            if "parameters" in img.info:  
                metadata = img.info["parameters"]  
            elif "prompt" in img.info:  
                metadata = img.info["prompt"]  
            # ComfyUI通常在"workflow"键中存储工作流  
            elif "workflow" in img.info:  
                metadata = img.info["workflow"]  
            # 有些版本可能使用"ComfyUI"键  
            elif "ComfyUI" in img.info:  
                metadata = img.info["ComfyUI"]  
            else:  
                # 尝试获取所有可用的元数据  
                metadata = json.dumps(img.info)  
            
            return metadata  
        except Exception as e:  
            print(f"Error extracting metadata: {e}")  
            return "{}"  
        

class RandomImageLoaderTlant:
    """
    ComfyUI自定义节点：从指定路径随机加载图片
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {
                    "default": "", 
                    "multiline": False,
                    "placeholder": "输入图片文件夹路径"
                }),
                "recursive": ("BOOLEAN", {
                    "default": True,
                    "label_on": "递归搜索",
                    "label_off": "仅当前目录"
                }),
                "is_fixed": ("BOOLEAN", {
                    "default": False,
                    "label_on": "固定图片",
                    "label_off": "每次随机"
                }),
            },
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "file_path")
    FUNCTION = "load_random_image"
    CATEGORY = "image/utils"
    
    @classmethod
    def IS_CHANGED(cls, path, recursive, is_fixed):
        """
        控制节点是否重新执行
        """
        if is_fixed:
            # 固定模式：返回固定值，避免重新执行
            return f"fixed_{path}_{recursive}"
        else:
            # 随机模式：返回时间戳，确保每次都重新执行
            return str(random.random()) 
    
    def get_image_files(self, path, recursive=True):
        """
        获取指定路径下的所有图片文件
        """
        if not os.path.exists(path):
            raise ValueError(f"路径不存在: {path}")
        
        if not os.path.isdir(path):
            raise ValueError(f"指定路径不是文件夹: {path}")
        
        # 支持的图片格式
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.gif'}
        image_files = []
        
        if recursive:
            # 递归搜索所有子文件夹
            for root, dirs, files in os.walk(path):
                for file in files:
                    if os.path.splitext(file.lower())[1] in image_extensions:
                        image_files.append(os.path.join(root, file))
        else:
            # 仅搜索当前目录
            for file in os.listdir(path):
                file_path = os.path.join(path, file)
                if os.path.isfile(file_path) and os.path.splitext(file.lower())[1] in image_extensions:
                    image_files.append(file_path)
        
        return image_files
    
    def load_image_to_tensor(self, image_path):
        """
        加载图片并转换为ComfyUI需要的tensor格式
        """
        try:
            # 使用PIL加载图片
            image = Image.open(image_path)
            
            # 转换为RGB模式（去除alpha通道）
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # 修正图片方向（根据EXIF信息）
            image = ImageOps.exif_transpose(image)
            
            # 转换为numpy数组
            image_np = np.array(image).astype(np.float32) / 255.0
            
            # 转换为torch tensor，并添加batch维度
            # ComfyUI期望的格式是 [batch, height, width, channels]
            image_tensor = torch.from_numpy(image_np)[None,]
            
            return image_tensor
            
        except Exception as e:
            raise ValueError(f"无法加载图片 {image_path}: {str(e)}")
    
    def load_random_image(self, path, recursive, is_fixed):
        """
        主要功能函数：随机加载图片
        """
        try:
            # 获取所有图片文件
            image_files = self.get_image_files(path, recursive)
            
            if not image_files:
                raise ValueError(f"在路径 {path} 中未找到任何图片文件")
            
            # 随机选择一个图片文件
            selected_image = random.choice(image_files)
            
            # 获取绝对路径
            absolute_path = os.path.abspath(selected_image)
            
            # 加载图片
            image_tensor = self.load_image_to_tensor(selected_image)
            
            mode = "固定图片" if is_fixed else "随机加载"
            print(f"{mode}: {absolute_path}")
            
            return (image_tensor, absolute_path)
            
        except Exception as e:
            # 创建一个错误占位图片
            error_image = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
            error_message = f"错误: {str(e)}"
            print(error_message)
            return (error_image, error_message)
        
class ReasoningLLMOutputCleaner:  
    """  
    A ComfyUI node to clean a string by removing <think>...</think> tags and stripping whitespace.  
    """  
    @classmethod  
    def INPUT_TYPES(cls):  
        """  
        Defines the input types for the node.  
        """  
        return {  
            "required": {  
                "text": ("STRING", {"multiline": True, "default": ""}),  
            }  
        }  

    RETURN_TYPES = ("STRING",)  
    RETURN_NAMES = ("cleaned_text",)  
    FUNCTION = "clean"  
    CATEGORY = "Gemini"  

    def clean(self, text):  
        """  
        The main function of the node.  
        It removes the content within <think></think> tags and strips surrounding whitespace.  
        """  
        # Use re.DOTALL to make '.' match newlines as well  
        # Use a non-greedy match '.*?' to handle multiple tags if they were to exist  
        cleaned_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)  
        
        # strip() removes leading/trailing whitespace, including newlines  
        cleaned_text = cleaned_text.strip()  
        
        return (cleaned_text,)  


class SaveImagePairForKontext:  
    """  
    一个ComfyUI节点，用于保存图像及其关联的文本文件。  
    该节点接收一张图像、保存路径、文件名、可选的前后缀以及文本内容。  
    它会将图像保存到指定位置，并根据文本内容是否存在，选择性地保存一个同名的.txt文件。  
    """  
    def __init__(self):  
        pass  

    @classmethod  
    def INPUT_TYPES(s):  
        """  
        定义节点的输入参数类型和属性。  
        """  
        return {  
            "required": {  
                "image": ("IMAGE", ),  
                "path": ("STRING", {"default": "D:/ComfyUI/output"}),  
                "filename": ("STRING", {"default": "image.png"}),  
                "prefix": ("STRING", {"default": ""}),  
                "suffix": ("STRING", {"default": ""}),  
                "text": ("STRING", {"multiline": True, "default": ""}),  
            },  
        }  

    RETURN_TYPES = ()  # 此节点不返回任何输出  
    FUNCTION = "save_image_pair"  
    OUTPUT_NODE = True # 标记为输出节点  
    CATEGORY = "image" # 在UI中的分类  

    def save_image_pair(self, image: torch.Tensor, path: str, filename: str, text: str, prefix: str = "", suffix: str = ""):  
        """  
        核心功能实现方法。  
        
        Args:  
            image (torch.Tensor): 输入的图像张量，格式为 (batch, height, width, channel)。  
            path (str): 图像和文本文件的保存目录。  
            filename (str): 基础文件名 (可以包含扩展名)。  
            text (str): 要写入 .txt 文件的内容。如果为空，则不保存txt文件。  
            prefix (str, optional): 文件名前缀. Defaults to "".  
            suffix (str, optional): 文件名后缀. Defaults to "".  
        """  
        # 检查并创建输出目录  
        if not os.path.exists(path):  
            print(f"路径 {path} 不存在，正在创建...")  
            os.makedirs(path, exist_ok=True)  

        # 分离基础文件名和扩展名  
        base_name, extension = os.path.splitext(filename)  
        
        # 处理图像批次  
        for i, single_image in enumerate(image):  
            # 构造最终的文件基础名 (不含扩展名)  
            # 如果是批处理，添加数字后缀  
            if image.shape[0] > 1:  
                current_base_name = f"{prefix}{base_name}{suffix}_{i:08d}"  
            else:  
                current_base_name = f"{prefix}{base_name}{suffix}"  

            # 构造完整的图像文件路径  
            image_file_path = os.path.join(path, f"{current_base_name}{extension}")  

            # 将Tensor图像转换为Pillow图像  
            # 1. 从Tensor转换到Numpy数组  
            # 2. 将像素值从 [0, 1] 范围转换到 [0, 255]  
            # 3. 转换为 uint8 类型  
            img_np = np.clip(255. * single_image.cpu().numpy(), 0, 255).astype(np.uint8)  
            
            # 4. 从Numpy数组创建Pillow Image对象  
            pil_image = Image.fromarray(img_np)  
            
            # 5. 保存图像  
            pil_image.save(image_file_path)  
            print(f"成功保存图片到: {image_file_path}")  

            # 如果文本内容不为空，则保存同名的txt文件  
            # 使用strip()来确保内容不只是空白字符  
            if text and text.strip():  
                text_file_path = os.path.join(path, f"{current_base_name}.txt")  
                try:  
                    with open(text_file_path, 'w', encoding='utf-8') as f:  
                        f.write(text)  
                    print(f"成功保存文本文件到: {text_file_path}")  
                except IOError as e:  
                    print(f"错误：无法写入文本文件 {text_file_path}: {e}")  

        return {} # 输出节点需要返回一个空字典  


class StringFormatterTlant:
    """
    A custom node for ComfyUI that formats a template string with multiple inputs.
    It replaces placeholders '{}' with provided parameters in order.
    It gracefully handles empty or missing parameters to prevent errors.
    """
    @classmethod
    def INPUT_TYPES(s):
        """
        Defines the input types for the node.
        """
        # Define up to 10 parameters. 
        # In the UI, users can right-click the node and select which optional inputs to show.
        # This is the standard ComfyUI way to handle a variable number of inputs without a dynamic "add" button.
        optional_params = {}
        for i in range(3, 11): # param3 to param10 are optional
            optional_params[f"param{i}"] = ("STRING", {"multiline": False, "default": ""})

        return {
            "required": {
                "template": ("STRING", {
                    "multiline": True,
                    "default": "A {adj} {noun} in the style of {artist}."
                }),
                "param1": ("STRING", {"multiline": False, "default": "beautiful"}),
                "param2": ("STRING", {"multiline": False, "default": "landscape"}),
            },
            "optional": optional_params
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("formatted_string",)
    FUNCTION = "format"
    CATEGORY = "utils"

    def format(self, template, **kwargs):
        """
        The main execution function of the node.
        
        Args:
            template (str): The string template with placeholders like {}.
            **kwargs: A dictionary containing all connected parameters (param1, param2, ...).
        
        Returns:
            A tuple containing the formatted string.
        """
        # Collect all parameters in the correct order (param1, param2, param3, ...)
        # We sort by the key to ensure the order is always correct.
        params = []
        # The number of maximum supported params is 10 as defined in INPUT_TYPES
        for i in range(1, 11):
            key = f"param{i}"
            if key in kwargs:
                # Handle potential None values from unconnected inputs, default to empty string
                value = kwargs[key]
                params.append(value if value is not None else "")

        # --- Exception Handling ---
        # Count the number of placeholders in the template
        num_placeholders = template.count('{}')

        # Pad the parameters list with empty strings if there are more placeholders than params.
        # This prevents an IndexError when format() is called.
        if len(params) < num_placeholders:
            padding_needed = num_placeholders - len(params)
            params.extend([""] * padding_needed)

        # The .format() method will automatically ignore extra parameters if there are
        # fewer placeholders than provided params, so we don't need to handle that case explicitly.

        try:
            # Perform the string formatting
            formatted_text = template.format(*params)
        except (IndexError, ValueError) as e:
            # Fallback in case of unexpected formatting errors
            print(f"\033[91m[StringFormatter] Error formatting string: {e}\033[0m")
            print(f"\033[91m[StringFormatter] Template: '{template}', Params: {params}\033[0m")
            # Return the original template as a safe fallback
            formatted_text = template

        # The node must return a tuple
        return (formatted_text,)


class LoadSpecificTxtFileTlant:  
    """  
    这是一个 ComfyUI 自定义节点，用于从指定的 .txt 文件路径加载文本内容。  
    """  
    
    @classmethod  
    def INPUT_TYPES(cls):  
        """  
        定义节点的输入类型。  
        - required: 定义了节点的必需输入。  
        - file_path: 一个字符串输入字段，用于指定文本文件的路径。  
                     "default" 设置了输入框中的默认提示文本。  
        """  
        return {  
            "required": {  
                "file_path": ("STRING", {  
                    "multiline": False, # 设置为单行输入  
                    "default": "C:\\path\\to\\your\\file.txt"  
                }),  
            },  
        }  

    # 定义节点的返回类型  
    RETURN_TYPES = ("STRING",)  
    
    # 定义节点的返回名称（可选，用于UI显示）  
    RETURN_NAMES = ("text",)  

    # 定义节点执行的主要功能  
    FUNCTION = "load_text_file"  

    # 定义节点在 ComfyUI 菜单中的分类  
    CATEGORY = "Tlant"  

    def load_text_file(self, file_path):  
        """  
        加载并返回文本文件的内容。  
        
        Args:  
            file_path (str): 用户在UI中输入的文本文件路径。  
            
        Returns:  
            tuple: 包含文件内容的元组 (text,)。如果文件不存在或读取失败，  
                   将返回一个包含错误信息的字符串。  
        """  
        # 检查文件路径是否存在并且确实是一个文件  
        if not os.path.isfile(file_path):  
            error_message = f"Error: File not found at the specified path: {file_path}"  
            print(error_message)  
            return (error_message,)  

        try:  
            # 使用 utf-8 编码打开并读取文件内容  
            with open(file_path, 'r', encoding='utf-8') as f:  
                text = f.read()  
            # ComfyUI 的函数返回值必须是一个元组  
            return (text,)  
        except Exception as e:  
            # 捕获其他可能的读取错误  
            error_message = f"Error reading file '{file_path}': {e}"  
            print(error_message)  
            return (error_message,)  


class LoadSequencedTxtFileTlant:  
    """  
    这是一个ComfyUI自定义节点，它按顺序从指定目录读取文本文件。  
    V2版本简化了输入，并增加了文件名作为输出。  

    功能：  
    - 遍历指定目录（可递归）以查找所有.txt文件。  
    - 对找到的文件名进行升序排序。  
    - 使用'txt_index'参数作为索引来选择文件，超过文件总数则取余。  
    - 输出所选文件的内容、完整路径以及不带路径的文件名。  
    """  
    @classmethod  
    def INPUT_TYPES(cls):  
        """  
        定义节点的输入参数。  
        """  
        return {  
            "required": {  
                "dir_path": ("STRING", {  
                    "default": "",  
                    "multiline": False,  
                    "placeholder": "Enter directory path"  
                }),  
                "txt_index": ("INT", {  
                    "default": 0,  
                    "min": 0,  
                    "max": 0xffffffffffffffff,  
                    "step": 1,  
                    "display": "number"  
                }),  
                "is_recursive": (["true", "false"], {  
                    "default": "false"  
                }),  
            },  
        }  

    # 定义节点的返回类型  
    RETURN_TYPES = ("STRING", "STRING", "STRING")  
    # 定义节点的返回名称  
    RETURN_NAMES = ("text", "txt_file_path", "txt_filename")  
    FUNCTION = "read_file_sequentially"  
    CATEGORY = "Tlant/File Operations"  

    def read_file_sequentially(self, dir_path, txt_index, is_recursive):  
        working_dir = dir_path  
        txt_files = []  
        is_recursive_bool = (is_recursive == "true")  

        try:  
            # 路径有效性验证  
            if not os.path.exists(working_dir):  
                return (f"Directory does not exist: {working_dir}", "", "")  
            if not os.path.isdir(working_dir):  
                return (f"Path is not a directory: {working_dir}", "", "")  

            # 文件收集逻辑  
            if is_recursive_bool:  
                for root, _, files in os.walk(working_dir):  
                    for file in files:  
                        if file.lower().endswith('.txt'):  
                            full_path = os.path.abspath(os.path.join(root, file))  
                            txt_files.append(full_path)  
            else:  
                for item in os.listdir(working_dir):  
                    full_path = os.path.abspath(os.path.join(working_dir, item))  
                    if os.path.isfile(full_path) and item.lower().endswith('.txt'):  
                        txt_files.append(full_path)  

            if not txt_files:  
                return ("No .txt files found in the specified directory.", "", "")  

        except PermissionError as e:  
            return (f"Permission denied: {str(e)}", "", "")  
        except Exception as e:  
            return (f"System error during file search: {str(e)}", "", "")  

        # 核心逻辑：排序并按索引选择  
        txt_files.sort()  # 按文件名升序排序  
        
        # 使用模运算（取余）来选择文件，实现循环  
        selected_index = txt_index % len(txt_files)  
        selected_file_path = txt_files[selected_index]  
        
        # 新增：获取不带路径的文件名  
        selected_filename = os.path.basename(selected_file_path)  

        # 文件读取逻辑，尝试多种编码  
        try:  
            content = ""  
            for encoding in ['utf-8', 'gbk', 'latin-1']:  
                try:  
                    with open(selected_file_path, 'r', encoding=encoding) as f:  
                        content = f.read()  
                    # 成功读取后，返回内容、完整路径和纯文件名  
                    return (content, selected_file_path, selected_filename)  
                except UnicodeDecodeError:  
                    continue # 编码不匹配，尝试下一种  
            
            # 如果所有编码都失败  
            error_msg = f"Failed to decode file '{selected_filename}' with common encodings."  
            return (error_msg, selected_file_path, selected_filename)  

        except Exception as e:  
            return (f"File read error: {str(e)}", selected_file_path, selected_filename)  


class OpenRouterApiTlantV1:
    """
    ComfyUI node for calling OpenRouter API with built-in caching support.
    Only re-executes when inputs actually change; use seed to force refresh.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": ("STRING", {
                    "default": "xiaomi/mimo-v2-flash",
                    "multiline": False,
                    "tooltip": "OpenRouter model identifier, e.g. google/gemini-2.5-pro-preview"
                }),
                "base_url": ("STRING", {
                    "default": "https://openrouter.ai/api/v1/chat/completions",
                    "multiline": False,
                    "tooltip": "API endpoint URL"
                }),
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "OpenRouter API key (sk-or-v1-...)"
                }),
                "system_prompt": ("STRING", {
                    "default": "You are a helpful assistant.",
                    "multiline": True,
                    "tooltip": "System prompt to set model behavior"
                }),
                "user_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "User prompt / instruction"
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "Controls randomness. Lower = more deterministic"
                }),
                "max_tokens": ("INT", {
                    "default": 2048,
                    "min": 1,
                    "max": 128000,
                    "step": 64,
                    "tooltip": "Maximum number of tokens in the response"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xFFFFFFFFFFFFFFFF,
                    "tooltip": "Change seed to force re-execution; keep fixed to use cache"
                }),
                "remove_think_tags": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Remove <think>...</think> blocks from output"
                }),
                "proxy_url": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Optional proxy. e.g. http://127.0.0.1:1080 or socks5://127.0.0.1:1080"
                }),
                "top_p": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Nucleus sampling threshold"
                }),
                "timeout": ("INT", {
                    "default": 300,
                    "min": 10,
                    "max": 1200,
                    "step": 10,
                    "tooltip": "Request timeout in seconds"
                }),
                "image_detail": (["auto", "low", "high"], {
                    "default": "auto",
                    "tooltip": "Image detail level for vision models"
                }),
            },
            "optional": {
                "images": ("IMAGE", {
                    "tooltip": "Optional image input for vision models. Supports batch."
                }),
                "user_prompt_input": ("STRING", {
                    "forceInput": True,
                    "tooltip": "External user prompt input, will be appended to user_prompt"
                }),
                "system_prompt_input": ("STRING", {
                    "forceInput": True,
                    "tooltip": "External system prompt input, will be appended to system_prompt"
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("response_text", "reasoning_content",)
    OUTPUT_TOOLTIPS = (
        "The model's text response",
        "Reasoning content returned by reasoning models (if available)",
    )
    FUNCTION = "execute"
    CATEGORY = "Tlant/API"
    DESCRIPTION = "Call OpenRouter API with automatic caching. Same inputs + same seed = skip API call."

    # ------------------------------------------------------------------ #
    #                        Main execution                               #
    # ------------------------------------------------------------------ #
    def execute(
        self,
        model_name,
        base_url,
        api_key,
        system_prompt,
        user_prompt,
        temperature,
        max_tokens,
        seed,
        remove_think_tags,
        proxy_url="",
        images=None,
        user_prompt_input=None,
        system_prompt_input=None,
        top_p=1.0,
        timeout=300,
        image_detail="auto",
    ):
        # ---- 合并 prompt ---- #
        final_system = self._merge_text(system_prompt, system_prompt_input)
        final_user = self._merge_text(user_prompt, user_prompt_input)

        if not final_user.strip() and images is None:
            return ("", "",)

        # ---- 构建 messages ---- #
        messages = []
        if final_system.strip():
            messages.append({"role": "system", "content": final_system})

        user_content = self._build_user_content(final_user, images, image_detail)
        messages.append({"role": "user", "content": user_content})

        # ---- 构建请求 ---- #
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/comfyanonymous/ComfyUI",
            "X-Title": "ComfyUI-OpenRouter-Tlant",
        }

        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
        }

        # ---- 代理设置 ---- #
        proxies = None
        if proxy_url and proxy_url.strip():
            p = proxy_url.strip()
            proxies = {"http": p, "https": p}

        # ---- 发送请求 ---- #
        try:
            print(f"[OpenRouterTlant] Calling model: {model_name}")
            resp = requests.post(
                url=base_url.strip(),
                headers=headers,
                json=payload,
                proxies=proxies,
                timeout=timeout,
            )
            resp.raise_for_status()
            result = resp.json()
        except requests.exceptions.Timeout:
            error_msg = f"[OpenRouterTlant] Request timed out after {timeout}s"
            print(error_msg)
            return (error_msg, "",)
        except requests.exceptions.RequestException as e:
            error_msg = f"[OpenRouterTlant] Request failed: {e}"
            # 尝试提取 API 返回的错误信息
            try:
                err_body = resp.json()
                error_msg += f"\nAPI response: {json.dumps(err_body, ensure_ascii=False, indent=2)}"
            except Exception:
                pass
            print(error_msg)
            return (error_msg, "",)

        # ---- 解析响应 ---- #
        response_text, reasoning_content = self._parse_response(result)

        # ---- 打印用量信息 ---- #
        usage = result.get("usage", {})
        if usage:
            print(f"[OpenRouterTlant] Tokens  prompt: {usage.get('prompt_tokens', '?')}  "
                  f"completion: {usage.get('completion_tokens', '?')}  "
                  f"total: {usage.get('total_tokens', '?')}")

        # ---- 移除 think 标签 ---- #
        if remove_think_tags and response_text:
            response_text = self._strip_think_tags(response_text)

        return (response_text, reasoning_content,)

    # ------------------------------------------------------------------ #
    #                     IS_CHANGED — 缓存控制核心                        #
    # ------------------------------------------------------------------ #
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """
        使用 **kwargs 接收所有参数，避免因 optional 参数缺失导致签名不匹配。
        """
        h = hashlib.sha256()

        try:
            for key in sorted(kwargs.keys()):
                val = kwargs[key]
                if key == "images" and val is not None:
                    # 图片用 shape + sum 做指纹
                    h.update(f"images_shape={val.shape}".encode("utf-8"))
                    h.update(f"images_sum={val.sum().item()}".encode("utf-8"))
                else:
                    h.update(f"{key}={val}".encode("utf-8"))

            result = h.hexdigest()
            print(f"[OpenRouterTlant] IS_CHANGED hash: {result}")
            print(f"[OpenRouterTlant] IS_CHANGED keys: {sorted(kwargs.keys())}")
            return result

        except Exception as e:
            print(f"[OpenRouterTlant] IS_CHANGED EXCEPTION: {e}")
            import traceback
            traceback.print_exc()
            # 出异常就返回固定值，避免重复执行
            return "error_fallback_fixed"

    # ------------------------------------------------------------------ #
    #                          Helper methods                             #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _merge_text(base_text: str, extra_text: str | None) -> str:
        """合并两段文本"""
        parts = []
        if base_text and base_text.strip():
            parts.append(base_text)
        if extra_text and extra_text.strip():
            parts.append(extra_text)
        return "\n".join(parts) if parts else ""

    @staticmethod
    def _image_to_base64(img_tensor) -> str:
        """将单张 ComfyUI IMAGE tensor (H,W,C float32 0~1) 转为 base64 PNG"""
        img_np = (img_tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np)
        buffer = io.BytesIO()
        img_pil.save(buffer, format="PNG", optimize=True)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _build_user_content(self, text: str, images, image_detail: str):
        """构建 user message 的 content 字段"""
        if images is None:
            return text

        # 多模态内容：先放图片，再放文本
        content_parts = []
        for i in range(images.shape[0]):
            img_b64 = self._image_to_base64(images[i])
            content_parts.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img_b64}",
                    "detail": image_detail,
                }
            })
        if text.strip():
            content_parts.append({"type": "text", "text": text})
        return content_parts

    @staticmethod
    def _parse_response(result: dict) -> tuple[str, str]:
        """从 API 响应中提取文本和推理内容"""
        choices = result.get("choices", [])
        if not choices:
            error_info = result.get("error", {})
            if error_info:
                return (f"API Error: {json.dumps(error_info, ensure_ascii=False)}", "")
            return ("No response from API", "")

        message = choices[0].get("message", {})
        response_text = message.get("content", "") or ""
        reasoning_content = message.get("reasoning_content", "") or ""

        # 某些模型把推理放在 <think> 标签里而非 reasoning_content 字段
        # 如果 reasoning_content 为空，尝试从 response_text 中提取
        if not reasoning_content:
            think_match = re.search(
                r'<think(?:ing)?>(.*?)</think(?:ing)?>',
                response_text,
                flags=re.DOTALL
            )
            if think_match:
                reasoning_content = think_match.group(1).strip()

        return (response_text, reasoning_content)

    @staticmethod
    def _strip_think_tags(text: str) -> str:
        """移除 <think>...</think> 和 <thinking>...</thinking> 标签及内容"""
        cleaned = re.sub(
            r'<think(?:ing)?>\s*.*?\s*</think(?:ing)?>',
            '',
            text,
            flags=re.DOTALL
        )
        # 清理多余的空行
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        return cleaned.strip()




# Define node mappings for ComfyUI  
NODE_CLASS_MAPPINGS = {  
    "OllamaPromptsGeneratorTlant": OllamaPromptsGeneratorTlant,
    "LoadRandomTxtFileTlant": LoadRandomTxtFileTlant,
    "LoadRandomTxtFileTlantV2": LoadRandomTxtFileTlantV2,
    "LoadRandomTxtFileTlantV3": LoadRandomTxtFileTlantV3,
    "OllamaSimpleTextGeneratorTlant": OllamaSimpleTextGeneratorTlant,
    "LoadImageAndExtractMetadataTlant": LoadImageAndExtractMetadataTlant,
    "RandomImageLoaderTlant": RandomImageLoaderTlant,
    "ReasoningLLMOutputCleaner": ReasoningLLMOutputCleaner,
    "SaveImagePairForKontext": SaveImagePairForKontext,
    "StringFormatterTlant": StringFormatterTlant,
    "LoadSpecificTxtFileTlant": LoadSpecificTxtFileTlant,
    "LoadSequencedTxtFileTlant": LoadSequencedTxtFileTlant,
    "OpenRouterApiTlantV1": OpenRouterApiTlantV1
}  

# Define display name for the node  
NODE_DISPLAY_NAME_MAPPINGS = {  
    "OllamaPromptsGeneratorTlant": "Ollama Prompts Generator Tlant",
    "LoadRandomTxtFileTlant": "Load Random Txt File Tlant",
    "LoadRandomTxtFileTlantV2": "Load Random Txt File Tlant V2",
    "LoadRandomTxtFileTlantV3": "Load Random Text File V3",
    "OllamaSimpleTextGeneratorTlant": "Ollama Simple Text Generator Tlant",
    "LoadImageAndExtractMetadataTlant": "Load Image & Extract Metadata",
    "RandomImageLoaderTlant": "Random Image Loader Tlant",
    "ReasoningLLMOutputCleaner": "Reasoning LLM Output Cleaner",
    "SaveImagePairForKontext": "Save Image Pair Text",
    "StringFormatterTlant": "String Formatter Tlant",
    "LoadSpecificTxtFileTlant": "Load Specific Txt File Tlant",
    "LoadSequencedTxtFileTlant": "Load Sequenced Txt File Tlant",
    "OpenRouterApiTlantV1": "OpenRouter API (Tlant V1)"
}

WEB_DIRECTORY = "./web"  