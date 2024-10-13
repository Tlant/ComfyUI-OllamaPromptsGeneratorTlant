import requests

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

    def execute(self, positive_text, reference_text, negative_text, prompt_template, custom_model, ollama_url):  

        prompt = prompt_template.format(positive_text=positive_text, reference_text=reference_text, negative_text=negative_text)

        payload = {"model": custom_model, "prompt": prompt, "stream": False}

        response = requests.post(ollama_url, json=payload)
        response.raise_for_status()
        result = response.json()["response"]
        return (result,)  


# Define node mappings for ComfyUI  
NODE_CLASS_MAPPINGS = {  
    "OllamaPromptsGeneratorTlant": OllamaPromptsGeneratorTlant  
}  

# Define display name for the node  
NODE_DISPLAY_NAME_MAPPINGS = {  
    "OllamaPromptsGeneratorTlant": "Ollama Prompts Generator Tlant"  
}