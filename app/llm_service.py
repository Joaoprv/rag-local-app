import logging
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain_huggingface import HuggingFacePipeline
from models.models import ModelType
from typing import Optional
import torch

logger = logging.getLogger(__name__)

class LLMService:
    """Service for loading and managing language models."""
    
    def load_model(self, model_type: ModelType) -> Optional[HuggingFacePipeline]:
        """Load the specified language model."""
        try:
            model_name = model_type.value
            logger.info(f"Loading model: {model_name}")
            
            tokenizer, model = self._load_base_model(model_name)
            pipe = self._create_pipeline(model, tokenizer)
            
            logger.info(f"Successfully loaded model: {model_name}")
            return HuggingFacePipeline(pipeline=pipe)
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}")
            return None
    
    def _load_base_model(self, model_name: str):
        """Load tokenizer and model with common configuration."""
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        return tokenizer, model
    
    def _create_pipeline(self, model, tokenizer):
        """Create text generation pipeline with standard configuration."""
        return pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.01,
            pad_token_id=tokenizer.eos_token_id
        )
