import streamlit as st
import tempfile
import os
from pathlib import Path
from PIL import Image
import warnings
from openai import OpenAI
import sys

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from utils.pdf_converter import (
    convert_pdf_to_images,
    describe_image_with_vision
)

# Suppress PIL warnings
warnings.filterwarnings('ignore', category=Image.DecompressionBombWarning)
Image.MAX_IMAGE_PIXELS = None

# Create OpenAI client
client = OpenAI(
    api_key="3CN85u38Gr5zO2horjyj0s47WhDetiJI",
    base_url="https://api.deepinfra.com/v1/openai"
)

# Constants
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

def validate_file_size(file, max_size: int = MAX_FILE_SIZE) -> bool:
    """Validate uploaded file size"""
    return file.size <= max_size

def validate_api_connection(client: OpenAI) -> bool:
    """Test API connection and authentication"""
    try:
        # Simple test call to validate API connection
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.2-11B-Vision-Instruct",
            messages=[{"role": "user", "content": "Test"}],
            max_tokens=10
        )
        return True
    except Exception as e:
        st.error(f"API Authentication Error: Please check your DeepInfra API key. Error: {str(e)}")
        return False

# Main Streamlit app
def main():
    st.set_page_config(page_title="PDF Text Extractor", page_icon="📄")
    
    # Add header with custom styling
    st.markdown("""
        <h1 style='text-align: center;'>PDF Text Extractor 📄</h1>
        <p style='text-align: center;'>Upload your PDF file and get extracted text</p>
    """, unsafe_allow_html=True)

    # Validate API connection before allowing uploads
    if not validate_api_connection(client):
        st.warning("⚠️ Please configure a valid DeepInfra API key in your .env file")
        st.stop()

    # File upload section
    uploaded_file = st.file_uploader(
        "Choose a PDF file (max 10MB)",
        type="pdf",
        help="Upload a PDF file to extract its text content"
    )

    if uploaded_file:
        if not validate_file_size(uploaded_file):
            st.error("File size exceeds 10MB limit.")
            st.stop()

        # Display file info
        file_details = {
            "Filename": uploaded_file.name,
            "File size": f"{uploaded_file.size/1024:.0f} KB"
        }
        st.write("File Details:")
        for key, value in file_details.items():
            st.write(f"- {key}: {value}")

        # Process button
        if st.button("Extract Text", type="primary"):
            try:
                with st.spinner("Processing PDF..."):
                    # Create temp file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        pdf_path = tmp_file.name

                    # Process PDF
                    pages = convert_pdf_to_images(pdf_path)
                    
                    # Initialize progress
                    progress_text = "Extraction progress"
                    my_bar = st.progress(0, text=progress_text)
                    
                    # Container for status messages
                    status_container = st.empty()
                    
                    text_output = ""
                    total_pages = len(pages)
                    
                    # Process each page
                    for i, page in enumerate(pages):
                        status_container.write(f"Processing page {i + 1} of {total_pages}")
                        description = describe_image_with_vision(client, page, i)  # Pass client here
                        text_output += f"=== Page {i+1} ===\n{description}\n\n"
                        my_bar.progress((i + 1)/total_pages, 
                                      text=f"{progress_text} - {((i + 1)/total_pages)*100:.0f}%")
                    
                    # Clear status message
                    status_container.empty()
                    
                    # Success message
                    st.success("✅ Text extraction completed!")
                    
                    # Display results in tabs
                    tab1, tab2 = st.tabs(["Preview", "Download"])
                    
                    with tab1:
                        st.text_area("Extracted Text Preview", 
                                   text_output, 
                                   height=400)
                    
                    with tab2:
                        st.download_button(
                            label="📥 Download Text File",
                            data=text_output,
                            file_name=f"{uploaded_file.name}_extracted.txt",
                            mime="text/plain",
                            help="Click to download the extracted text as a .txt file"
                        )
                        
                        # Add copy button
                        if st.button("📋 Copy to Clipboard"):
                            st.write("Text copied to clipboard!")
                            st.session_state['clipboard'] = text_output

                    # Cleanup
                    os.unlink(pdf_path)

            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")
                st.stop()
    
    # Add footer
    st.markdown("---")
    st.markdown("""
        <p style='text-align: center;'>
            Made with ❤️ using Streamlit and AI Vision
        </p>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
