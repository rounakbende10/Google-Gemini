# MultiLanguage Invoice Extractor

This repository contains a Streamlit web application for a MultiLanguage Invoice Extractor. The application leverages the Google GenerativeAI Gem API to generate responses based on user prompts and uploaded invoice images.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

# Set up environment variables:
1. Create a .env file.

2. Add your Google API key:

```bash
 GOOGLE_API_KEY=your_api_key
```

# Run the application:
 
```bash
 streamlit run your_script_name.py
```

## Usage
1. Input Prompt:
 * Enter a prompt describing the inquiry about the invoice.
2. Upload Invoice Image:
 * Choose an image file (JPEG, JPG, PNG) of the invoice.
3. Click "Tell me about the invoice" button to generate and display responses.

## Features
1. Utilizes Google GenerativeAI Gem API for content generation.
2. Supports multiple image formats for invoice uploads.
3. Provides responses based on user prompts and uploaded invoice images.

## Note
Ensure proper configuration of environment variables and API key for seamless functionality.


