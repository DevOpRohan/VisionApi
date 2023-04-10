import openai
import torch
import os
from fastapi import FastAPI, Request, Depends
from fastapi.exceptions import HTTPException
from pydantic import BaseModel

import todo_model
from visual_services import Vision, download_image, save_and_process_image
from todo_model import get_user_id_by_ip, create_user_with_ip
from db_service import handle_database_interaction
from config import OPENAI_API_KEY


# Create the 'image' folder if it doesn't exist
os.makedirs('image', exist_ok=True)

# Visual Services
vision  = Vision()


app = FastAPI()

openai.api_key = OPENAI_API_KEY

VISION_PROMPT = "You are \"Vision,\" a multidimensional (able to understand languages and images) and multilingual AI designed to engage and assist visually impaired people. \nIf necessary, you can activate the following services by returning the corresponding keywords as response.\nServices-Keyword Map:\n{\n1. Live Object Locator -> \"@name_of_object\"\n2. Visual Questions -> \"@vq:<question>\"\n3. To-Do Services -> \"@Todo:<query>\"\n4. Closing App -> \"@exit\"\n}\n=====\nRules:\n{\n1. Only return corresponding keyword to use a service. \ne.g @pen, @Todo: To buy grocery and @vq:What is colour of this wall .\n2. Don't expose the service keywords.\n3. Try to respond in user's language but, Services-Keyword should be in English.\n4. Limit the use of object locator, use it only to locate/find objects live but in other case use @vq:<ques>\n}"

chat_history_general = {}
chat_history_db_service = {}
ip_user_id_cache = {}  # Memory cache for IP-User ID mapping
user_id_vq_cache = {}  # Memory cache for User ID-Visual Question mapping

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

    elif ai_response.startswith("@vq:"):
        history.pop(0)
        question = ai_response[4:].strip()
        # Set the question in id-question map user_id_vq_cache = {}
        user_id_vq_cache[userId] = question

        # Return the @vq to take the image
        return "@vq"

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

    # Get the question from the user_id_vq_cache
    ques = user_id_vq_cache[userId]
    image_path = download_image(image_url, user_id)
    processed_image_path = save_and_process_image(image_path, user_id)

    # Use Vision to get the answer of complex queries
    answer = vision.get_answer(ques, processed_image_path)

    # Return the answer
    return answer

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)