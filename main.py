from app.interface import demo
from logging_config import setup_logging
import logging

if __name__ == "__main__":
    setup_logging(log_level="INFO")
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 Starting RAG PDF Chat Assistant...")
    logger.info("📚 Loading interface...")
    
    try:
        demo.launch(
            share=False,
            debug=True,
            server_name="0.0.0.0",
            server_port=7860,
            show_error=True,
            quiet=False
        )
    except Exception as e:
        logger.error(f"Failed to launch application: {e}")
        raise