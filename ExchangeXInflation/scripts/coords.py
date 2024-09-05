import fitz  # PyMuPDF

# Path to your PDF file
pdf_file = 'rate.pdf'

# Open the PDF file
pdf_document = fitz.open(pdf_file)

for page_num in range(len(pdf_document)):
    page = pdf_document.load_page(page_num)
    text_blocks = page.get_text("blocks")

    print(f"Page {page_num + 1}:")

    for block in text_blocks:
        bbox = block[:4]  # Bounding box coordinates
        text = block[4]  # Text content
        print(f"Block coordinates: {bbox}, Text: {text}")

# Close the PDF document
pdf_document.close()
