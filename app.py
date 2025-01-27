import os
import base64
import tempfile
import warnings
from io import BytesIO
from pathlib import Path
from typing import List, Any
from PIL import Image
import streamlit as st
import fitz  # PyMuPDF
from openai import OpenAI

# Load environment variables from Streamlit secrets
DEEPINFRA_API_KEY = st.secrets["DEEPINFRA_API_KEY"]
MODEL_NAME = st.secrets.get("MODEL_NAME", "meta-llama/Llama-3.2-11B-Vision-Instruct")
API_BASE_URL = st.secrets.get("API_BASE_URL", "https://api.deepinfra.com/v1/openai")

# Suppress PIL warnings
warnings.filterwarnings('ignore', category=Image.DecompressionBombWarning)
Image.MAX_IMAGE_PIXELS = None

# Constants for image processing
MAX_IMAGE_SIZE = 1500  # Reduced maximum dimension in pixels
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Initialize OpenAI client
client = OpenAI(api_key=DEEPINFRA_API_KEY, base_url=API_BASE_URL)

# Helper Functions
def resize_image(image: Image.Image, max_size: int = MAX_IMAGE_SIZE) -> Image.Image:
    """Resize image while maintaining aspect ratio"""
    ratio = min(max_size / float(image.size[0]), max_size / float(image.size[1]))
    if ratio < 1:
        new_size = tuple(int(dim * ratio) for dim in image.size)
        return image.resize(new_size, Image.Resampling.LANCZOS)
    return image

def convert_pdf_to_images(pdf_path: str) -> List[Image.Image]:
    """Convert PDF pages to images using PyMuPDF"""
    pdf_document = fitz.open(pdf_path)
    images = []
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(resize_image(img))
    pdf_document.close()
    return images

def encode_image(image: Image.Image) -> str:
    """Convert image to base64 string"""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def describe_image_with_vision(client: OpenAI, image: Image.Image, page_num: int) -> str:
    """Send image to Vision LLM for text extraction"""
    try:
        instructions = """You are a text extraction tool. Your ONLY task is to extract ALL text from this document EXACTLY as it appears, with special attention to headers and tables. Follow these STRICT rules (STRICTLY DO NOT INCLUDE ANY OF THOSE RULES IN YOUR RESPONSE):

1. **Headers and Page Information**:
   - Extract headers at the top of pages, including page numbers and metadata.
   - Preserve header formatting and position.

2. **Table Handling**:
   - Extract ALL table content cell by cell, preserving structure using tabs or spaces.
   - Include table captions, headers, and footnotes.

3. **Formatting and Structure**:
   - Maintain line breaks, spacing, and alignment.
   - Preserve bold, italics, underline, and hierarchical headings.

4. **Exact Text Extraction**:
   - Do NOT summarize, interpret, or modify any content.
   - Include all text exactly as it appears, marking unclear text as [UNREADABLE].

5. **Special Elements**:
   - Mark merged cells in tables, rotated text, and complex formatting explicitly.

6. **Blank Pages**:
   - If a page is blank, return: "[NO TEXT FOUND]"

Remember: Accuracy in headers and tables is CRITICAL. Extract EVERYTHING exactly as it appears."""

        encoded_image = encode_image(image)
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": f"Extract text from page {page_num + 1} of the document."},
                {"role": "user", "content": f"data:image/jpeg;base64,{encoded_image}"}
            ],
            max_tokens=4096
        )

        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[ERROR] Failed to process page {page_num + 1}: {str(e)}"

# Main Streamlit App
def main():
    st.set_page_config(page_title="PDF Text Extractor", page_icon="📄")

    st.title("PDF Text Extractor 📄")
    st.write("Upload your PDF file to extract its text.")

    uploaded_file = st.file_uploader("Choose a PDF file (max 10MB)", type="pdf")

    if uploaded_file:
        if uploaded_file.size > MAX_FILE_SIZE:
            st.error("File size exceeds 10MB limit.")
            return

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(uploaded_file.getvalue())
            temp_pdf_path = temp_pdf.name

        try:
            pages = convert_pdf_to_images(temp_pdf_path)
            if not pages:
                st.error("No pages could be extracted from the PDF.")
                return

            extracted_text = []
            for i, page in enumerate(pages):
                st.write(f"Processing page {i + 1} of {len(pages)}...")
                text = describe_image_with_vision(client, page, i)
                extracted_text.append(f"=== Page {i + 1} ===\n{text}")

            final_text = "\n\n".join(extracted_text)
            st.success("Text extraction completed!")
            st.download_button(
                label="Download Extracted Text",
                data=final_text,
                file_name="extracted_text.txt",
                mime="text/plain"
            )

        except Exception as e:
            st.error(f"Error: {e}")
        finally:
            os.unlink(temp_pdf_path)

if __name__ == "__main__":
    main()

