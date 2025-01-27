# PDF Text Extractor

A Streamlit application that extracts text from PDF documents using AI vision models.

## Setup

1. Create a Streamlit secrets file (.streamlit/secrets.toml) with:
```toml
DEEPINFRA_API_KEY = "your-api-key"
MODEL_NAME = "meta-llama/Llama-3.2-11B-Vision-Instruct"
API_BASE_URL = "https://api.deepinfra.com/v1/openai"
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the app:
```bash
streamlit run app.py
```

## Requirements
- Python 3.8+
- Poppler (for pdf2image)
- DeepInfra API key
