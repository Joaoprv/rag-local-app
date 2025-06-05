from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from transformers import pipeline
import logging
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextSummarizer:
    """Modern text summarizer using LangChain with LCEL (LangChain Expression Language)."""
    
    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        """Initialize the summarizer with a specified model.
        
        Args:
            model_name: HuggingFace model name for summarization
        """
        self.model_name = model_name
        self.llm = self._setup_llm()
        self.chain = self._create_chain()
    
    def _setup_llm(self) -> HuggingFacePipeline:
        """Setup the HuggingFace pipeline and LLM."""
        try:
            summarization_pipeline = pipeline(
                "summarization", 
                model=self.model_name
            )
            return HuggingFacePipeline(
                pipeline=summarization_pipeline,
                model_kwargs={"max_length": 150, "min_length": 30, "do_sample": False}
            )
        except Exception as e:
            logger.error(f"Failed to setup LLM: {e}")
            raise
    
    def _create_chain(self):
        """Create the modern LangChain chain using LCEL."""
        prompt = PromptTemplate(
            input_variables=["text"],
            template="Summarize the following text clearly and concisely:\n\n{text}\n\nSummary:"
        )
        
        # Modern LCEL chain composition
        chain = (
            {"text": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        return chain
    
    def summarize(self, text: str) -> Optional[str]:
        """Summarize the given text.
        
        Args:
            text: Text to be summarized
            
        Returns:
            Summarized text or None if error occurs
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for summarization")
            return None
            
        try:
            logger.info("Starting text summarization...")
            result = self.chain.invoke(text)
            logger.info("Summarization completed successfully")
            return result
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            return None




if __name__ == "__main__":
    
    sample_text = """
    Artificial intelligence has revolutionized various sectors of the global economy. 
    From automating industrial processes to developing virtual assistants, AI is 
    transforming how we work and live. In healthcare, machine learning algorithms 
    help with early disease diagnosis. In education, adaptive systems personalize 
    learning for each student. The future promises even more innovations with the 
    continuous development of this technology. Companies are investing billions 
    in AI research, and governments are creating policies to regulate its use 
    while promoting innovation.
    """
    
    try:
        summarizer = TextSummarizer()
        summary = summarizer.summarize(sample_text)
        
        if summary:
            print("=" * 50)
            print("ORIGINAL TEXT:")
            print("=" * 50)
            print(sample_text.strip())
            print("\n" + "=" * 50)
            print("SUMMARY:")
            print("=" * 50)
            print(summary.strip())
        else:
            print("Failed to generate summary.")
            
    except Exception as e:
        logger.error(f"Application error: {e}")