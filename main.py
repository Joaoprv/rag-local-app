import pymupdf

def extract_text_from_pdf(pdf_path):
    doc = pymupdf.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

if __name__ == "__main__":
    pdf_path = "../data/dummy.pdf"
    extracted_text = extract_text_from_pdf(pdf_path)
    print(extracted_text)