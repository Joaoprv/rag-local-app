import logging
import fitz
from app.summarizer import summarize
from app.chat_rag import build_vector_store, get_chat_chain

logger = logging.getLogger(__name__)

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file with logging."""
    try:
        logger.info(f"Opening PDF file: {pdf_path}")
        doc = fitz.open(pdf_path)
        text = ""
        
        logger.info(f"PDF has {len(doc)} pages")
        
        for page_num, page in enumerate(doc):
            page_text = page.get_text()
            text += page_text
            logger.debug(f"Extracted {len(page_text)} characters from page {page_num + 1}")
        
        doc.close()
        logger.info(f"Successfully extracted {len(text)} total characters from PDF")
        return text
        
    except Exception as e:
        logger.error(f"Error extracting text from PDF {pdf_path}: {e}")
        return ""


def process_pdf(file, app_state):
    """Process PDF using the AppState object."""
    if app_state.llm is None:
        logger.warning("PDF processing attempted without loaded model")
        return "❌ Please load a model first!"
    
    if file is None:
        logger.warning("PDF processing attempted without file") 
        return "❌ Please upload a PDF file!"
    
    try:
        logger.info(f"Processing PDF: {file.name}")
        
        text = extract_text_from_pdf(file.name)
        if not text.strip():
            logger.warning(f"No text extracted from PDF: {file.name}")
            return "❌ No text could be extracted from the PDF!"
        
        logger.info(f"Extracted {len(text)} characters from PDF")

        logger.info("Generating document summary")
        summary = summarize(text, app_state.llm)
        
        logger.info("Building vector store")
        vectorstore = build_vector_store(text)
        logger.info("Creating chat chain")
        app_state.chat_chain = get_chat_chain(app_state.llm, vectorstore)
        app_state.document_processed = True
        
        logger.info("PDF processing completed successfully")
        return f"✅ PDF processed successfully!\n\n📄 Summary:\n{summary}"
        
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        return f"❌ Error processing PDF: {str(e)}"