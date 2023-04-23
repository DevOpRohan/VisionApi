# VisionAPI
- The Vision API is designed to assist visually impaired users by providing a multimodal and multilingual AI-powered system that can answer general questions, perform live object detection, and handle complex visual queries. It is accessible through both Android and Web clients.
- For Persistance/Authentication userId-Ip Map is used.
i.e On Ip-Change, the user will be treated as a new user.
## Android Client

For the Android client, users interact with two main API endpoints: `Vision` and `uploadImage`.

1. **Vision Endpoint**: The Vision API can respond in several ways:
   - General answers to user queries.
   - `@name_of_object`: Triggers live object detection.
   - `@vq`: Triggers complex visual queries. Android will capture an image and upload it to the `uploadImage` API to obtain the answer.
   - `@exit`: Closes the app.

2. **uploadImage Endpoint**: This endpoint is used to upload images for complex visual queries. Once the image is uploaded, the API processes and returns the answer to the query.

## Web Client

For the Web client, users interact with the `VisionWeb` endpoint.

1. **VisionWeb Endpoint**: Similar to the Android client, the VisionWeb API can respond with general answers, complex visual queries, and app exit commands. When the API returns `@vq` for a complex visual query, a modal will pop up on the web app, allowing users to either upload an image or provide an image link for processing.

In both Android and Web clients, the Vision API offers a seamless experience for visually impaired users, accommodating their needs by providing live object detection, complex visual queries, and general assistance.
## Requirements
- Docker
- GPT-4 API key and other api keys(mentioned in .ENV file)
- At least 12GB of RAM if using CPU only. If using a GPU, at least 4GB of GPU memory and 8GB of RAM are required.

## Setup with PostgreSQL Database
1. Clone this repo:
    ```bash
    git clone https://github.com/DevOpRohan/VisionApi.git
    ```
2. Go to the cloned directory:
    ```bash
    cd VisionApi
    ```
3. Add the required API keys to the `.env` file:
    - `OPENAI_API_KEY`: Your OpenAI API key. Visit this [link](https://platform.openai.com/) to create an account and get the API key.
    - `OCR_API_KEY`: Your OCR API key. Visit this [link](https://ocr.space/ocrapi) to create an account and get the API key.
    - `DATABASE_URL`: The connection URL to your PostgreSQL database. Visit this [link](https://www.digitalocean.com/try/managed-databases-postgresql) to set up a PostgreSQL database on Digital Ocean.
    - `NGROK_AUTH_TOKEN`: Ngrok Auth token. Visit this [link](https://dashboard.ngrok.com/get-started/your-authtoken) to set up and obtain your ngrok auth token.

4. Build the Docker image:
    ```bash
    docker build -t visionapi .
    ```
5. Run the Docker image:
    ```bash
    docker run -it -p 8000:8000 visionapi
    ```
6. The API is now running and publicly available via an ngrok tunnel. To use it, set up the [VisionAndroid](https://github.com/DevOpRohan/VisionAndroid.git) app.

## Setup with SQLite
1. Clone this repo:
    ```bash
    git lfs install
    git clone https://huggingface.co/spaces/devoprohan/VisionAPi
    ```
2. Follow steps 2-6 from the PostgreSQL setup above. You won't need to worry about setting up a PostgreSQL database, as an in-memory SQLite database will be used.

## Note
After running for the first time, errors may occur due to required Python libraries needing a restart. Simply restart the Docker container, and it should work correctly.