import os
import base64
import tempfile
import warnings
import shutil
from io import BytesIO
from pathlib import Path
from typing import List, Any
from PIL import Image
import streamlit as st
from pdf2image import convert_from_path
from openai import OpenAI
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

# Load environment variables from Streamlit secrets
DEEPINFRA_API_KEY = st.secrets["DEEPINFRA_API_KEY"]
MODEL_NAME = st.secrets.get("MODEL_NAME", "meta-llama/Llama-3.2-11B-Vision-Instruct")
API_BASE_URL = st.secrets.get("API_BASE_URL", "https://api.deepinfra.com/v1/openai")

# Constants from pdf_converter1.py
MAX_IMAGE_SIZE = 2000
DPI = 200
MAX_WIDTH = 1700
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Suppress PIL warnings
warnings.filterwarnings('ignore', category=Image.DecompressionBombWarning)
Image.MAX_IMAGE_PIXELS = None

# Initialize OpenAI client with proper configuration
try:
    client = OpenAI(
        api_key=DEEPINFRA_API_KEY,
        base_url=API_BASE_URL,
        http_client=httpx.Client(
            timeout=60.0,
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
            follow_redirects=True
        ),
        max_retries=3
    )
except Exception as e:
    st.error(f"Failed to initialize API client: {str(e)}")
    st.stop()

# Helper Functions from pdf_converter1.py
def resize_image(image: Image.Image, max_size: int = MAX_IMAGE_SIZE) -> Image.Image:
    """Resize image while maintaining aspect ratio"""
    if not isinstance(image, Image.Image):
        raise ValueError("Input must be a PIL Image object")
    
    try:
        ratio = min(max_size/float(image.size[0]), max_size/float(image.size[1]))
        if ratio < 1:
            new_size = tuple(int(dim * ratio) for dim in image.size)
            return image.resize(new_size, Image.Resampling.LANCZOS)
        return image
    except Exception as e:
        raise Exception(f"Image resize failed: {str(e)}")

def convert_pdf_to_images(pdf_path: str) -> List[Image.Image]:
    """Convert PDF pages to images with size optimization"""
    if not Path(pdf_path).exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    try:
        pages = convert_from_path(
            pdf_path,
            dpi=DPI,
            size=(MAX_WIDTH, None)
        )
        return [resize_image(page) for page in pages]
    except Exception as e:
        raise Exception(f"PDF conversion failed: {str(e)}")

def encode_image(image: Any) -> str:
    """Convert image to base64 string"""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

# Add retry decorator for API calls
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
def describe_image_with_vision(client: OpenAI, image: Any, page_num: int) -> str:
    """Send image to vision model for text extraction"""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{
                "role": "system",
                "content": "You are a text extraction tool. Extract ALL text EXACTLY as shown in the image. Do NOT explain, interpret, or provide instructions. Only output the exact text found in the image."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """Extract ALL text from this document EXACTLY as it appears, with special attention to these rules:

1. Extract EVERY SINGLE character, word, number, and symbol exactly as shown
2. Preserve ALL formatting, including:
   - Line breaks and spacing
   - Tables and their structure
   - Headers and footers
   - Page numbers and dates
   - Lists and bullet points
   - Special characters and symbols

3. For tables:
   - Keep exact cell contents
   - Maintain column alignment
   - Use ASCII characters for borders (│, ─, ┌, ┐, └, ┘)
   - Preserve headers and labels
   - Keep numerical data exactly as shown

4. Mark special cases:
   - [UNREADABLE] for unclear text
   - [MERGED] for merged table cells
   - [ROTATED] for vertical text
   - [NO TEXT FOUND] for blank pages

5. DO NOT:
   - Add any explanations or descriptions
   - Skip any text, even if seems unimportant
   - Rearrange or reorganize content
   - Summarize or paraphrase
   - Make assumptions about unclear text

Extract the text exactly as it appears in the document:"""
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encode_image(image)}"
                        }
                    }
                ]
            }],
            max_tokens=8192,
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        raise Exception(f"API Error: {str(e)}")

def check_dependencies():
    """Check if required system dependencies are installed"""
    try:
        # First try using pdf2image directly
        pages = convert_from_path(
            BytesIO(b"%PDF-1.7\n"),  # Minimal valid PDF
            dpi=72,
            size=(100, None)
        )
        return True
    except Exception as e:
        if "poppler" in str(e).lower():
            st.error("""
            ⚠️ Poppler is not installed. The app will attempt to install it automatically.
            If this persists, try manually installing:
            - Ubuntu/Debian: sudo apt-get install poppler-utils
            - MacOS: brew install poppler
            - Windows: included in python-poppler-binary
            """)
        else:
            st.error(f"Dependency check failed: {str(e)}")
        return False

def main():
    try:
        # Validate API connection
        client.models.list()  # Simple API call to test connection
    except Exception as e:
        st.error("Failed to connect to API. Please check your credentials and connection.")
        st.stop()
        
    # Check dependencies first, before showing UI
    if not check_dependencies():
        st.stop()
    
    st.set_page_config(page_title="PDF Text Extractor", page_icon="📄")

    st.title("PDF Text Extractor 📄")
    st.write("Upload your PDF file to extract its text.")

    uploaded_file = st.file_uploader("Choose a PDF file (max 10MB)", type="pdf")

    if uploaded_file:
        if uploaded_file.size > MAX_FILE_SIZE:
            st.error("File size exceeds 10MB limit.")
            return

        progress_bar = st.progress(0)
        status_text = st.empty()

        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf:
            temp_pdf.write(uploaded_file.getvalue())
            temp_pdf_path = temp_pdf.name

        try:
            status_text.text("Converting PDF to images...")
            pages = convert_pdf_to_images(temp_pdf_path)
            total_pages = len(pages)

            if total_pages == 0:
                st.error("No pages found in PDF.")
                return

            extracted_texts = []
            for i, page in enumerate(pages):
                progress = (i + 1) / total_pages
                progress_bar.progress(progress)
                status_text.text(f"Processing page {i + 1} of {total_pages}...")
                
                text = describe_image_with_vision(client, page, i)
                extracted_texts.append(f"=== Page {i + 1} ===\n{text}")

            final_text = "\n\n".join(extracted_texts)
            progress_bar.progress(1.0)
            status_text.text("Processing complete!")

            st.success(f"Successfully processed {total_pages} pages!")

            # Preview section
            with st.expander("Preview extracted text", expanded=True):
                st.text_area("Extracted Text Preview", final_text, height=300)

            # Download button
            st.download_button(
                label="Download Extracted Text",
                data=final_text,
                file_name=f"extracted_{uploaded_file.name}.txt",
                mime="text/plain"
            )

        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.info("Please try again with a different PDF file or contact support.")

        finally:
            try:
                os.unlink(temp_pdf_path)
            except Exception:
                pass

if __name__ == "__main__":
    main()
