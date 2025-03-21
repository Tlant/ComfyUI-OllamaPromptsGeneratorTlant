import requests
import os
import random
import hashlib  
import json  
from server import PromptServer
from aiohttp import web 

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


# Define node mappings for ComfyUI  
NODE_CLASS_MAPPINGS = {  
    "OllamaPromptsGeneratorTlant": OllamaPromptsGeneratorTlant,
    "LoadRandomTxtFileTlant": LoadRandomTxtFileTlant,
    "LoadRandomTxtFileTlantV2": LoadRandomTxtFileTlantV2,
    "LoadRandomTxtFileTlantV3": LoadRandomTxtFileTlantV3,
    "OllamaSimpleTextGeneratorTlant": OllamaSimpleTextGeneratorTlant
}  

# Define display name for the node  
NODE_DISPLAY_NAME_MAPPINGS = {  
    "OllamaPromptsGeneratorTlant": "Ollama Prompts Generator Tlant",
    "LoadRandomTxtFileTlant": "Load Random Txt File Tlant",
    "LoadRandomTxtFileTlantV2": "Load Random Txt File Tlant V2",
    "LoadRandomTxtFileTlantV3": "Load Random Text File V3",
    "OllamaSimpleTextGeneratorTlant": "Ollama Simple Text Generator Tlant"
}

WEB_DIRECTORY = "./web"  