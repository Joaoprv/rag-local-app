import logging
from langchain.prompts import PromptTemplate
from prompts.summarizer_prompts import SUMMARIZER_PROMPTS, DEFAULT_LANGUAGE

logger = logging.getLogger(__name__)


def get_summarizer_prompt(language="português"):
    """
    Get a summarizer prompt template for the specified language.
    
    Args:
        language (str): Language for the prompt template
        
    Returns:
        PromptTemplate: Configured prompt template
    """
    language_key = language.lower()
    
    if language_key in ["português", "portuguese", "pt"]:
        language_key = "português"
    elif language_key in ["english", "en"]:
        language_key = "english"

    
    if language_key in SUMMARIZER_PROMPTS:
        prompt_config = SUMMARIZER_PROMPTS[language_key]
        logger.debug(f"Loading summarizer prompt for language: {language_key}")
    else:
        logger.warning(f"Language '{language}' not found, using default: {DEFAULT_LANGUAGE}")
        prompt_config = SUMMARIZER_PROMPTS[DEFAULT_LANGUAGE]
    
    return PromptTemplate(
        input_variables=prompt_config["input_variables"],
        template=prompt_config["template"]
    )


def list_available_languages():
    """
    Get list of available languages for summarizer prompts.
    
    Returns:
        list: Available language keys
    """
    return list(SUMMARIZER_PROMPTS.keys())