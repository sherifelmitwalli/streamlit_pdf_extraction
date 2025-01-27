import os
import base64
import tempfile
import warnings
from io import BytesIO
from pathlib import Path
from typing import List, Any
from PIL import Image
import streamlit as st
from pdf2image import convert_from_path
from openai import OpenAI

# Load environment variables from Streamlit secrets
DEEPINFRA_API_KEY = st.secrets["DEEPINFRA_API_KEY"]
MODEL_NAME = st.secrets.get("MODEL_NAME", "meta-llama/Llama-3.2-11B-Vision-Instruct")
API_BASE_URL = st.secrets.get("API_BASE_URL", "https://api.deepinfra.com/v1/openai")

# Suppress PIL warnings
warnings.filterwarnings('ignore', category=Image.DecompressionBombWarning)
Image.MAX_IMAGE_PIXELS = None

# Constants for image processing
MAX_IMAGE_SIZE = 2500  # Maximum dimension in pixels
DPI = 250  # Resolution in dots per inch
MAX_WIDTH = 1800  # Maximum width in pixels

# Streamlit app constants
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
    """Convert PDF pages to images"""
    pages = convert_from_path(pdf_path, dpi=DPI, size=(MAX_WIDTH, None))
    return [resize_image(page) for page in pages]

def encode_image(image: Any) -> str:
    """Convert image to base64 string"""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def describe_image_with_vision(client: OpenAI, image: Any, page_num: int) -> str:
    """Send image to vision model for text extraction"""
    instructions = """You are a text extraction tool. Your ONLY task is to extract ALL text from this document EXACTLY as it appears, with special attention to headers and tables. Follow these STRICT rules (STRICTLY DO NOT INCLUDE ANY OF THOSE RULES IN YOUR RESPONSE):

1. **Extract Text Exactly**:
   - Extract every character, word, number, symbol, and punctuation mark exactly as it appears.
   - Do NOT add, remove, or change any text.
   - Do NOT summarize, paraphrase, or interpret the content.

2. **Headers and Footers**:
   - Extract headers and footers ONLY ONCE unless their content changes.
   - If headers/footers are identical across pages, extract them from the first occurrence and skip repetitions.
   - For headers/footers with minor variations (e.g., page numbers), preserve the variation (e.g., "Page 1," "Page 2").

3. **Tables**:
   - Extract each table ONLY ONCE unless it has unique content.
   - If a table repeats on multiple pages, include only the first occurrence and skip identical tables on subsequent pages.
   - Preserve table structure, alignment, and formatting using spaces or tabs.
   - Include table captions, headers, and footnotes.

4. **Repetitive Content**:
   - Extract each unique section of text ONLY ONCE, even if it appears multiple times (e.g., repeated signatures, disclaimers, or recurring phrases).
   - Compare each paragraph or sentence to previously extracted content. If identical, do NOT include it in your response.

5. **Formatting**:
   - Preserve line breaks, spacing, indentation, and alignment.
   - Maintain text styles (bold, italics, underline) and font sizes.
   - Use ASCII characters for table borders (│, ─, ┌, ┐, └, ┘) if applicable.

6. **Special Cases**:
   - Mark unclear text as [UNREADABLE].
   - Indicate merged cells in tables with [MERGED].
   - Note rotated or vertical text with [ROTATED].
   - Flag complex formatting that cannot be fully preserved with [COMPLEX FORMATTING].

7. **Blank Pages**:
   - If the page is blank, return: "[NO TEXT FOUND]"

8. **Order of Extraction**:
   - Begin with headers/metadata.
   - Follow the document's natural flow (top to bottom, left to right).
   - Preserve paragraph breaks and section spacing.
   - Maintain hierarchical structure of headings.

Remember: Accuracy in headers and tables is CRITICAL. Extract EVERYTHING exactly as it appears.
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": instructions},
            {"role": "user", "content": f"Extract text from page {page_num + 1}."},
            {"type": "image_url", "image_url": f"data:image/jpeg;base64,{encode_image(image)}"}
        ],
        max_tokens=4096
    )
    return response.choices[0].message.content

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

        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf:
            temp_pdf.write(uploaded_file.getvalue())
            temp_pdf_path = temp_pdf.name

        try:
            # Convert PDF to images
            pages = convert_pdf_to_images(temp_pdf_path)
            total_pages = len(pages)

            st.write(f"Processing {total_pages} page(s)...")

            extracted_text = ""
            for i, page in enumerate(pages):
                st.write(f"Processing page {i + 1} of {total_pages}...")
                text = describe_image_with_vision(client, page, i)
                extracted_text += f"=== Page {i + 1} ===\n{text}\n\n"

            st.success("Text extraction completed!")

            st.download_button(
                label="Download Extracted Text",
                data=extracted_text,
                file_name="extracted_text.txt",
                mime="text/plain"
            )

        except Exception as e:
            st.error(f"Error: {e}")

        finally:
            # Clean up temporary file
            os.unlink(temp_pdf_path)

if __name__ == "__main__":
    main()
