import openai
import torch
import os
from fastapi import FastAPI, Request, Depends
from fastapi.exceptions import HTTPException
from pydantic import BaseModel

import todo_model
from visual_services import *
from todo_model import get_user_id_by_ip, create_user_with_ip
from db_service import handle_database_interaction
from config import OPENAI_API_KEY

#Visual Foundation Services
# Create the 'image' folder if it doesn't exist
os.makedirs('image', exist_ok=True)

# Load the models
device = "cuda" if torch.cuda.is_available() else "cpu"
image_captioning = ImageCaptioning(device=device)
visual_question_answering = VisualQuestionAnswering(device=device)
ocr = Ocr()


app = FastAPI()

openai.api_key = OPENAI_API_KEY

VISION_PROMPT = "====\nYou are \"Vision,\" a multidimensional (able to understand languages and images) and multilingual AI designed to engage and assist visually impaired people. \nIf necessary, you can activate the following services by returning the corresponding keywords as response.\nServices-Keyword Map:\n{\n1. Live Object Locator -> \"@name_of_object\"\n2. Scene Understanding -> \"@scene\"\n3. Visual Questions -> \"@vq:<question>\"\n4. Extract Information -> \"@ocr\"\n5. To-Do Services -> \"@Todo:<query>\"\n6. Closing App -> \"@exit\"\n}\n=====\nRules:\n{\n1. Only return corresponding keyword to use a service. \ne.g @pen, @ocr , @scene, @Todo: To buy grocery and @vq:What is colour of this wall .\n2. Don't expose the service keywords.\n3. Try to respond in user's language but, Services-Keyword should be in English.\n}"

chat_history_general = {}
chat_history_db_service = {}
ip_user_id_cache = {}  # Memory cache for IP-User ID mapping


@app.on_event("startup")
async def startup_event():
    await todo_model.create_tables()


@app.on_event("startup")
async def startup_event():
    result = await todo_model.get_all_tasks()


class Message(BaseModel):
    role: str
    content: str


class UploadImageInput(BaseModel):
    imageLink: str
    userId: str


async def get_user_id(request: Request):
    ip = request.client.host

    if ip in ip_user_id_cache:
        user_id = ip_user_id_cache[ip]
    else:
        user_id = await get_user_id_by_ip(ip)

        if not user_id:
            user_id = await create_user_with_ip(ip)
            print("New user created with id: ", user_id)

        ip_user_id_cache[ip] = user_id

    return user_id


@app.get("/vision")
async def vision(request: Request, q: str, userId: str = Depends(get_user_id)):
    ip = request.client.host

    if ip not in chat_history_general:
        chat_history_general[ip] = {"uuid": userId, "history": []}
        chat_history_db_service[ip] = {"active": False, "history": []}

    if chat_history_db_service[ip]["active"]:
        print("DB Service is already Active")
        db_response = await handle_database_interaction(userId, q, chat_history_db_service[ip])
        return db_response

    history = chat_history_general[ip]["history"]
    history.append({"role": "user", "content": q})

    preset = [
        {
            "role": "system",
            "content": VISION_PROMPT,
        },
        *history,
    ]

    completion = openai.ChatCompletion.create(
        model="gpt-4",
        messages=preset,
        temperature=0,
        max_tokens=512,
    )

    ai_response = completion.choices[0].message["content"]

    if ai_response.startswith("@Todo:") or chat_history_db_service[ip]["active"]:
        history.pop(0)
        todo_message = ai_response[6:].strip()
        db_response = await handle_database_interaction(userId, todo_message, chat_history_db_service[ip])
        return db_response
    else:
        history.append({"role": "assistant", "content": ai_response})

        if len(history) > 10:
            history.pop(0)
            history.pop(0)

        return ai_response


@app.get("/uploadImage")
async def upload_image(request: Request, imageLink: str, userId: str = Depends(get_user_id)):
    image_url = imageLink
    user_id = userId
    image_path = download_image(image_url, user_id)
    processed_image_path = save_and_process_image(image_path, user_id)
    # ocr_result = ocr.ocr_file(processed_image_path)
    ocr_result = ocr.ocr_url(image_url)
    return {"status": "success", "message": "Image uploaded and processed successfully", "ocrResult": ocr_result}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)