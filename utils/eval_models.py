import os
from typing import TypedDict, Any, Optional, List
from dotenv import load_dotenv
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from typing import Dict, Any, Optional
from azure.ai.projects.models import EvaluationTarget, AOAIModelConfig

"""
Module to evaluate & compare models using the Azure AI Projects SDK.
"""

class ModelTarget:
    """
    A target model for evaluation that wraps the Azure EvaluationTarget functionality
    but handles initialization differently to avoid Azure SDK internal attribute errors.
    """
    def __init__(self, model_name: str = "gpt-4o", system_message: str = "You are a helpful assistant.") -> None:
        """
        Initialize with the selected model name.
        """
        
        # Load environment variables
        load_dotenv(override=True)

        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        
        # Create a proper AOAIModelConfig
        self.model_config = AOAIModelConfig(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            azure_deployment=model_name
        )
        
        # Store the parameters directly instead of using EvaluationTarget
        self.system_message = system_message
        self.model_name = model_name
        
    def to_evaluation_target(self):
        """
        Create and return a properly initialized EvaluationTarget instance
        """
        return EvaluationTarget(
            system_message=self.system_message,
            model_config=self.model_config
        )