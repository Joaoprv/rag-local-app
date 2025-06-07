from app.interface import demo

if __name__ == "__main__":
    print("🚀 Starting RAG PDF Chat Assistant...")
    print("📚 Loading interface...")
    
    demo.launch(
        share=False,
        debug=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        quiet=False
    )