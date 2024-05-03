from typing import Dict, List, Union
from config import config

from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.chat_models import BedrockChat


class ChatModel:
    def __init__(self, model_name: str, model_kwargs: Dict):
        self.model_config = config["models"][model_name]
        self.model_id = self.model_config["model_id"]
        self.model_kwargs = model_kwargs
        self.llm = BedrockChat(model_id=self.model_id, model_kwargs=model_kwargs, streaming=True)

    def format_prompt(self, prompt: str) -> Union[str, List[Dict]]:
        """
        Format the input prompt according to the model's requirements.
        """
        model_config = self.model_config
        if model_config.get("input_format") == "text":
            # If input format is 'text', return the prompt as is
            return prompt
        elif model_config.get("input_format") == "list_of_dicts":
            # If input format is 'list_of_dicts', create a dictionary containing the prompt text and its type
            prompt_text = {"type": "text", "text": prompt}
            # Return the prompt as a list of dictionaries
            return [prompt_text]
        else:
            # If input format is neither 'text' nor 'list_of_dicts', raise a ValueError
            raise ValueError(f"Unsupported input format for model: {self.model_id}")


