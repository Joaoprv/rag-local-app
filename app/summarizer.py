import logging
from langchain_core.runnables import Runnable
from langchain.text_splitter import RecursiveCharacterTextSplitter
from prompts.loader import get_summarizer_prompt

logger = logging.getLogger(__name__)

def get_summarizer_chain(llm, language="português") -> Runnable:
    """Create a summarization chain with improved prompting."""
    
    logger.debug(f"Creating summarizer chain for language: {language}")
    
    prompt = get_summarizer_prompt(language)
    
    return prompt | llm


def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    """Split text into manageable chunks for processing."""
    logger.debug(f"Chunking text of length {len(text)} with chunk_size={chunk_size}, overlap={chunk_overlap}")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_text(text)
    logger.info(f"Text split into {len(chunks)} chunks")
    return chunks


def summarize(text, llm, language="português", max_chunk_size=2000):
    """
    Summarize text with support for long documents.
    
    Args:
        text (str): Text to summarize
        llm: Language model to use
        language (str): Language for the summary
        max_chunk_size (int): Maximum size of text chunks
    
    Returns:
        str: Generated summary
    """
    
    if not text.strip():
        logger.warning("Empty text provided for summarization")
        return "No text provided for summarization."
    
    logger.info(f"Starting summarization of text with {len(text)} characters")
    
    try:
        # If text is short enough, summarize directly
        if len(text) <= max_chunk_size:
            logger.debug("Text is short enough for direct summarization")
            chain = get_summarizer_chain(llm, language)
            result = chain.invoke({"text": text})
            
            # Handle different response formats
            if isinstance(result, dict):
                summary = result.get("text", str(result))
            else:
                summary = str(result)
            
            logger.info("Direct summarization completed successfully")
            return summary
        
        # For longer texts, chunk and summarize
        logger.info("Text is too long, using chunked summarization")
        chunks = chunk_text(text, chunk_size=max_chunk_size//2)
        chain = get_summarizer_chain(llm, language)
        
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            logger.info(f"Summarizing chunk {i+1}/{len(chunks)}")
            try:
                result = chain.invoke({"text": chunk})
                if isinstance(result, dict):
                    summary = result.get("text", str(result))
                else:
                    summary = str(result)
                chunk_summaries.append(summary)
                logger.debug(f"Chunk {i+1} summarized successfully")
            except Exception as e:
                logger.error(f"Error summarizing chunk {i+1}: {e}")
                chunk_summaries.append(f"[Error processing chunk {i+1}]")
        
        # Combine chunk summaries
        combined_summary = "\n\n".join(chunk_summaries)
        logger.info("Combined chunk summaries")
        
        # If combined summary is still too long, summarize it again
        if len(combined_summary) > max_chunk_size:
            logger.info("Combined summary too long, performing final summarization")
            final_result = chain.invoke({"text": combined_summary})
            if isinstance(final_result, dict):
                final_summary = final_result.get("text", str(final_result))
            else:
                final_summary = str(final_result)
            logger.info("Final summarization completed")
            return final_summary
        
        logger.info("Chunked summarization completed successfully")
        return combined_summary
        
    except Exception as e:
        logger.error(f"Error in summarization: {e}")
        return f"Error generating summary: {str(e)}"