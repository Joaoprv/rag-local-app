from dataclasses import dataclass
from typing import Optional, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class ModelType(Enum):
    FALCON = "tiiuae/falcon-rw-1b"
    LLAMA = "meta-llama/llama-2-7b-chat-hf"

@dataclass
class AppState:
    llm: Optional[Any] = None
    chat_chain: Optional[Any] = None
    current_model: Optional[ModelType] = None
    document_processed: bool = False
    
    def reset_chat_chain(self):
        """Reset chat chain when model changes or new document is processed."""
        self.chat_chain = None
        self.document_processed= False