import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Usage
pdf_path = "operating_systems_three_easy_pieces.pdf"
book_text = extract_text_from_pdf(pdf_path)

# Save it to a .txt file for later use
with open("three_pieces_of_os.txt", "w", encoding="utf-8") as f:
    f.write(book_text)

print("✅ Text extracted and saved.")
