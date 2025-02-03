import requests
import os
import random

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
        每次执行时强制节点重新运行，确保随机选择文件。  
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

# Define node mappings for ComfyUI  
NODE_CLASS_MAPPINGS = {  
    "OllamaPromptsGeneratorTlant": OllamaPromptsGeneratorTlant,
    "LoadRandomTxtFileTlant": LoadRandomTxtFileTlant
}  

# Define display name for the node  
NODE_DISPLAY_NAME_MAPPINGS = {  
    "OllamaPromptsGeneratorTlant": "Ollama Prompts Generator Tlant",
    "LoadRandomTxtFileTlant": "Load Random Txt File Tlant"  
}