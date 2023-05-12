# VisionAPI

VisionAPI is an AI-powered system designed to assist visually impaired users with a multimodal and multilingual solution. It can answer general questions, perform live object detection, and handle complex visual queries. The system is accessible through both Android and Web clients.

## Features

- Provides general answers to user queries
- Triggers live object detection when the API returns `@name_of_object`
- Initiates complex visual queries when the API returns `@vq`
- Exits the app when the API returns `@exit`
- User interaction through Android and Web clients
- Persistence and authentication via userId-IP mapping (treats user as new on IP change)

## Android Client

The Android client interacts with two primary API endpoints: `Vision` and `uploadImage`.

1. **Vision Endpoint**: The Vision API can provide the following responses:
   - General answers to user queries
   - Triggers live object detection when the API returns `@name_of_object`
   - Initiates complex visual queries when the API returns `@vq`. The Android client will then capture and upload an image to the `uploadImage` API to obtain the answer.
   - Exits the app when the API returns `@exit`

2. **uploadImage Endpoint**: This endpoint is responsible for uploading images for complex visual queries. The API processes the uploaded image and returns the answer to the query.

## Web Client

The Web client interacts with the `VisionWeb` endpoint.

1. **VisionWeb Endpoint**: Similar to the Android client, the VisionWeb API can provide the following responses:
   - General answers to user queries
   - Initiates complex visual queries when the API returns `@vq`. A modal will appear on the web app, allowing users to either upload an image or provide an image link for processing.
   - Exits the app when the API returns `@exit`

Both the Android and Web clients offer a seamless experience for visually impaired users by accommodating their needs with live object detection, complex visual queries, and general assistance.

## Requirements

- Docker
- GPT-4 API key and other API keys (mentioned in the .ENV file)
- At least 12GB of RAM for CPU-only usage. If using a GPU, at least 4GB of GPU memory and 8GB of RAM are required.

## Default Setup
In the default setup, an in-memory SQLite database is used for data persistence.

1. Clone the repository:
   ```
   git clone https://github.com/DevOpRohan/VisionApi.git
   ```
2. Change to the cloned directory:
   ```
   cd VisionApi
   ```
3. Add the required API keys to the `.env` file:
   - `OPENAI_API_KEY`: Your OpenAI API key. Visit [OpenAI](https://platform.openai.com/) to create an account and obtain the API key.
   - `OCR_API_KEY`: Your OCR API key. Visit [OCR.space](https://ocr.space/ocrapi) to create an account and obtain the API key.
   - `NGROK_AUTH_TOKEN`: Ngrok Auth token. Visit [ngrok](https://dashboard.ngrok.com/get-started/your-authtoken) to set up and obtain your ngrok auth token.

4. Build the Docker image:
   ```
   docker build -t visionapi .
   ```
5. Run the Docker image:
   ```
   docker run -it -p 7860:7860 visionapi
   ```
6. The API is now running and publicly accessible via an ngrok tunnel. To use it, set up the [VisionAndroid](https://github.com/DevOpRohan/VisionAndroid.git) app.

## Setup with PostgreSQL
In the default setup, an in-memory SQLite database is used. To set up with PostgreSQL, follow the "Setup with PostgreSQL" instructions below.
1. Follow steps 1-3 from the Default Setup section.
2. In the '.env' file, set `DATABASE_URL` to your PostgreSQL database connection URL (visit [Digital Ocean](https://www.digitalocean.com/try/managed-databases-postgresql) to set up a PostgreSQL database).
3. Modify the Database related code:
   - In `todo_model.py`, comment out the code from lines 88 to 158 and uncomment the code from lines 1 to 87 (remove lines 2 and 87 as they are within triple quotes).
   - In `db_service.py`, update the `handle_db_interaction` function by changing this line:
     ```python
     temp = sys_prompt.format(SQL=sqlite_schema, userId=user_id, currDate=current_date);
     ```
     to
     ```python
     temp = sys_prompt.format(SQL=sqlite_schema, userId=user_id, currDate=current_date, ip=ip);
     ```

## Note

After running for the first time, errors may occur due to required Python libraries needing a restart. Simply restart the Docker container, and it should work correctly.