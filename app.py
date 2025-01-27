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
MAX_IMAGE_SIZE = 2500  # Maximum dimension in pixels
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
    try:
        with st.spinner("Opening PDF file..."):
            pdf_document = fitz.open(pdf_path)
            total_pages = len(pdf_document)
            st.info(f"Found {total_pages} pages in PDF")
            
            images = []
            progress_bar = st.progress(0)
            
            for page_num in range(total_pages):
                progress = (page_num + 1) / total_pages
                progress_bar.progress(progress)
                
                page = pdf_document[page_num]
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                images.append(resize_image(img))
                
            pdf_document.close()
            return images
    except ImportError:
        st.error("Error: PyMuPDF (fitz) is not properly installed. Please check your installation.")
        st.stop()
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        st.info("If the error persists, try a different PDF file or contact support.")
        raise

def encode_image(image: Any) -> str:
    """Convert image to base64 string"""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def describe_image_with_vision(client: OpenAI, image: Any, page_num: int) -> str:
    """Send image to vision model for text extraction"""
    instructions = """You are a text extraction tool. Your ONLY task is to extract ALL text from this document EXACTLY as it appears, with special attention to headers and tables. Follow these STRICT rules (STRICTLY DO NOT INCLUDE ANY OF THOSE RULES IN YOUR RESPONSE):

1. **Headers and Page Information**:
   - Always extract headers at the top of pages
   - Include page numbers, dates, or any other metadata
   - Preserve header formatting and position
   - Extract running headers and footers

2. **Table Handling**:
   - Extract ALL table content cell by cell
   - Maintain table structure using tabs or spaces
   - Preserve column headers and row labels
   - Keep numerical data exactly as shown
   - Include table borders and separators using ASCII characters
   - Format multi-line cells accurately

3. **Exact Text Only**: Extract every character, word, number, symbol, and punctuation mark exactly as it appears. Do NOT:
   - Add any text not present in the document
   - Remove any text present in the document
   - Change any text present in the document
   - Include any commentary, analysis, or interpretation

4. **Preserve Formatting**: Maintain the exact:
   - Line breaks and spacing
   - Indentation and alignment
   - Text styles (bold, italics, underline)
   - Font sizes and styles
   - Page layout and structure

5. **Order and Structure**:
   - Begin with page headers/metadata
   - Follow the document's natural flow
   - Extract text in reading order (top to bottom, left to right)
   - Preserve paragraph breaks and section spacing
   - Maintain hierarchical structure of headings

6. **Table-Specific Output Format**:
   - Use consistent spacing for columns
   - Align numerical data properly
   - Preserve column widths where possible
   - Use ASCII characters for table borders (│, ─, ┌, ┐, └, ┘)
   - Include table captions and notes

7. **Special Elements**:
   - Mark footnotes and endnotes appropriately
   - Preserve bullet points and numbered lists
   - Include figure captions and references
   - Extract sidebar content in position

8. **Clarity Rules**:
   - Mark unclear text as [UNREADABLE]
   - Indicate merged cells in tables
   - Note rotated or vertical text
   - Flag complex formatting that can't be fully preserved

9. **Strict Prohibitions**: Do NOT:
   - Summarize or paraphrase
   - Analyze or interpret content
   - Rearrange table data
   - Skip any text, even if it seems irrelevant
   - Add explanations or descriptions
   - Make assumptions about unclear content

10. **Verification**: If the page is blank, return: "[NO TEXT FOUND]"

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

        progress_text = st.empty()
        progress_text.text("Preparing PDF processing...")

        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf:
            temp_pdf.write(uploaded_file.getvalue())
            temp_pdf_path = temp_pdf.name

        try:
            # Convert PDF to images
            pages = convert_pdf_to_images(temp_pdf_path)
            
            if not pages:
                st.error("No pages could be extracted from the PDF.")
                return

            extracted_text = []
            progress_bar = st.progress(0)
            
            for i, page in enumerate(pages):
                progress = (i + 1) / len(pages)
                progress_bar.progress(progress)
                progress_text.text(f"Processing page {i + 1} of {len(pages)}...")
                
                text = describe_image_with_vision(client, page, i)
                extracted_text.append(f"=== Page {i + 1} ===\n{text}")

            final_text = "\n\n".join(extracted_text)
            progress_text.text("Processing complete!")
            progress_bar.progress(1.0)

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
            # Clean up temporary file
            os.unlink(temp_pdf_path)

if __name__ == "__main__":
    main()
