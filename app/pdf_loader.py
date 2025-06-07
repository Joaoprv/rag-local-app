
import logging
import fitz

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

if __name__ == "__main__":
    # Setup basic logging for standalone execution
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    pdf_path = "data/dummy.pdf"
    logger.info(f"Testing PDF extraction with: {pdf_path}")
    extracted_text = extract_text_from_pdf(pdf_path)
    logger.info(f"Extracted text length: {len(extracted_text)}")
    logger.debug(f"Extracted text preview: {extracted_text[:200]}...")