# MultiLanguage Invoice Extractor

This repository contains a Streamlit web application for a MultiLanguage Invoice Extractor. The application leverages the Google GenerativeAI Gem API to generate responses based on user prompts and uploaded invoice images.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt

# Set up environment variables:

 ```Create a .env file.
# Add your Google API key:

``` GOOGLE_API_KEY=your_api_key
 
``` streamlit run your_script_name.py

Usage
Input Prompt:

Enter a prompt describing the inquiry about the invoice.
Upload Invoice Image:

Choose an image file (JPEG, JPG, PNG) of the invoice.
Click "Tell me about the invoice" button to generate and display responses.

Features
Utilizes Google GenerativeAI Gem API for content generation.
Supports multiple image formats for invoice uploads.
Provides responses based on user prompts and uploaded invoice images.
Note
Ensure proper configuration of environment variables and API key for seamless functionality.


