# VisionAPI Setup on LINUX

## Requirements
- Python
- GPT-4
- At least 12GB of RAM if using CPU only. If using a GPU, at least 8GB of GPU memory and 8GB of RAM are required.

## Setup
1. Clone this repo
2. Create a `.env` file in the root directory of the project and add the following environment variables:
    - `OPENAI_API_KEY`: Your OpenAI API key
    - `OCR_API_KEY`: Your OCR API key
    - `DATABASE_URL`: The URL to your database

3. Install the dependencies by running `pip install -r requirements.txt` in the root directory of the project.

4. Run the file `main.py