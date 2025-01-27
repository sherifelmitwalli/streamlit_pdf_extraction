# PDF to Text Extractor 📄

A Streamlit application that uses advanced vision models (Llama-3.2-11B-Vision) to accurately extract text from PDF files.

## Features
- 🔍 PDF to text conversion using state-of-the-art vision models
- 📱 User-friendly Streamlit interface
- 📊 Real-time progress tracking
- 💾 Downloadable text output
- 📋 Copy to clipboard functionality
- 🛠️ Configurable model parameters
- 🔒 Secure API key management

## Project Structure

```
pdf-to-text/
├── src/
│   ├── app.py               # Main Streamlit application
│   ├── utils/
│   │   ├── __init__.py     # Package initialization
│   │   └── pdf_converter.py # PDF processing utilities
│   └── config/
│       └── settings.py      # Application settings
├── requirements.txt         # Project dependencies
├── .env                    # Environment variables
└── README.md              # Documentation
```

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd pdf-to-text
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root with the following content:
   ```env
   DEEPINFRA_API_KEY=your_api_key_here
   MODEL_NAME=meta-llama/Llama-3.2-11B-Vision-Instruct
   API_BASE_URL=https://api.deepinfra.com/v1/openai
   TEMPERATURE=0.7
   MAX_TOKENS=4096
   ```

## Usage

1. Start the Streamlit application:
   ```bash
   streamlit run src/app.py
   ```

2. Access the application in your web browser (typically at `http://localhost:8501`)

3. Upload a PDF file (max 10MB)

4. Click "Extract Text" to process the file

5. View the extracted text in the Preview tab or download it as a text file

## Configuration

The application can be configured through the `.env` file:

- `DEEPINFRA_API_KEY`: Your DeepInfra API key (required)
- `MODEL_NAME`: Vision model to use (default: meta-llama/Llama-3.2-11B-Vision-Instruct)
- `API_BASE_URL`: DeepInfra API endpoint
- `TEMPERATURE`: Model temperature for text generation (default: 0.7)
- `MAX_TOKENS`: Maximum tokens per request (default: 4096)

## Error Handling

If you encounter any errors:

1. **API Authentication Error**:
   - Ensure your DeepInfra API key is correctly set in the `.env` file
   - Verify the API key is valid and has necessary permissions

2. **File Size Error**:
   - Check that your PDF file is under 10MB
   - Consider splitting larger files

3. **Processing Error**:
   - Ensure the PDF is not corrupted
   - Check if the PDF is password protected

## Performance Tips

- Use PDFs with clear, high-quality text for best results
- Avoid PDFs with complex layouts if possible
- For large PDFs, expect longer processing times

## Contributing

Contributions are welcome! Feel free to:
- Submit bug reports
- Propose new features
- Create pull requests
- Improve documentation
