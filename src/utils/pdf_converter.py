from pdf2image import convert_from_path
from io import BytesIO
import base64
from openai import OpenAI
from typing import List, Any, Tuple
from PIL import Image
import warnings
from pathlib import Path
from src.config.settings import settings

# Suppress PIL warnings and set image processing configs
warnings.filterwarnings('ignore', category=Image.DecompressionBombWarning)
Image.MAX_IMAGE_PIXELS = None

# Image processing constants
MAX_IMAGE_SIZE = 2000
DPI = 200
MAX_WIDTH = 1700

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

def describe_image_with_vision(client: OpenAI, image: Any, page_num: int) -> str:
    """Send image to vision model for text extraction"""
    try:
        response = client.chat.completions.create(
            model=settings.MODEL_NAME,  # Use model name from settings
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """You are a text extraction tool. Your ONLY task is to extract ALL text from this document EXACTLY as it appears, with special attention to headers and tables. Follow these STRICT rules:

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

Here is the document:"""
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encode_image(image)}"
                        }
                    }
                ]
            }],
            max_tokens=settings.MAX_TOKENS,
            temperature=settings.TEMPERATURE
        )
        return response.choices[0].message.content
    except Exception as e:
        raise Exception(f"API Error: {str(e)}")
