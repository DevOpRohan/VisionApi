import shutil
import json
import openai
import torch
import os

from langchain import PromptTemplate
from pyngrok import ngrok, conf
from pydantic import BaseModel
from datetime import datetime

from fastapi import FastAPI, Request, Depends, UploadFile, File
from fastapi.exceptions import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import todo_model
from OpenAIAPI import OpenAIAPI
from utils import parse_action_line
from visual_services import Vision, download_image, save_and_process_image
from todo_model import get_user_id_by_ip, create_user_with_ip
from db_service import handle_database_interaction
from config import OPENAI_API_KEY, NGROK_AUTH_TOKEN
from prompt import vis_sys_prompt, init_prompt_android, init_prompt_web

# Create the 'image' folder if it doesn't exist
os.makedirs('image', exist_ok=True)

# Visual Services
vis = Vision()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)



openai_api = OpenAIAPI(OPENAI_API_KEY)





chat_history_general = {}
chat_history_db_service = {}
ip_user_id_cache = {}  # Memory cache for IP-User ID mapping
user_id_vq_cache = {}  # Memory cache for User ID-Visual Question mapping


@app.on_event("startup")
async def startup_event():
    await todo_model.create_tables()
    # Set the ngrok authentication token
    # auth_token = NGROK_AUTH_TOKEN
    # ngrok_config = conf.PyngrokConfig(auth_token=auth_token)
    # conf.set_default(ngrok_config)
    #
    # # Start the ngrok tunnel to make the api public on internet
    # ngrok_tunnel = ngrok.connect(7860)
    # print('Public URL:', ngrok_tunnel.public_url)


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
    print(q)
    ip = request.client.host

    if ip not in chat_history_general:
        chat_history_general[ip] = {"uuid": userId, "history": []}
        chat_history_db_service[ip] = {"active": False, "history": []}

    if chat_history_db_service[ip]["active"]:
        print("DB Service is already Active")
        db_response = await handle_database_interaction(userId, q, chat_history_db_service[ip])
        return JSONResponse(content={"message": db_response})

    history = chat_history_general[ip]["history"]
    temp = init_prompt_android.format(userQuery=q)
    print(temp)
    history.append({"role": "user", "content": temp})

    preset = [
        {
            "role": "system",
            "content": vis_sys_prompt,
        },
        *history,
    ]

    completion = await openai_api.chat_completion(
        model="gpt-4",
        messages=preset,
        temperature=0,
        max_tokens=512,
    )

    ai_response = completion.choices[0].message["content"]
    print(ai_response)
    action = parse_action_line(ai_response)
    print(action)
    history.pop(0)

    if action.startswith("@answer:"):
        return JSONResponse(content={"message": action[8:].strip()})

    elif action.startswith("@error:"):
        return JSONResponse(content={"message": action[7:].strip()})

    elif action.startswith("@todo:") or chat_history_db_service[ip]["active"]:
        db_response = await handle_database_interaction(userId, q, chat_history_db_service[ip])
        return JSONResponse(content={"message": db_response})

    elif action.startswith("@vq:"):
        user_id_vq_cache[userId] = q
        return JSONResponse(content={"message": "@vq"})

    else:
        return JSONResponse(content={"message": ai_response})



@app.get("/uploadImageLink")
async def upload_image(request: Request, imageLink: str, userId: str = Depends(get_user_id)):
    image_url = imageLink.strip('"')
    current_time = datetime.now()
    time_stamp = current_time.strftime("%Y%m%d%H%M%S%f")

    # Get the question from the user_id_vq_cache
    ques = user_id_vq_cache[userId]
    image_path = download_image(image_url, time_stamp)
    processed_image_path = save_and_process_image(image_path, time_stamp)

    # Use Vision to get the answer of complex queries
    answer = vis.get_answer(ques, processed_image_path)

    # Return the answer
    return JSONResponse(content={"message": answer})


@app.post("/uploadImage")
async def image_upload(request: Request, image: UploadFile = File(...), userId: str = Depends(get_user_id)):
    user_id = userId

    # Save the uploaded image to a temporary file
    temp_image_path = f"image/temp_{user_id}_{image.filename}"
    with open(temp_image_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    # Get the question from the user_id_vq_cache
    ques = user_id_vq_cache[userId]

    processed_image_path = save_and_process_image(temp_image_path, user_id)

    # Assuming vis is an instance of your Vision class
    # Use Vision to get the answer of complex queries
    answer = vis.get_answer(ques, processed_image_path)

    # Remove the temporary image file
    os.remove(temp_image_path)
    os.remove(processed_image_path)

    print(answer)

    # Return the answer in JSON format
    return JSONResponse(content={"message": answer})


# WEB API
@app.get("/visionweb")
async def visionweb(request: Request, q: str, userId: str = Depends(get_user_id)):
    print(q)
    ip = request.client.host

    if ip not in chat_history_general:
        chat_history_general[ip] = {"uuid": userId, "history": []}
        chat_history_db_service[ip] = {"active": False, "history": []}

    if chat_history_db_service[ip]["active"]:
        print("DB Service is already Active")
        db_response = await handle_database_interaction(userId, q, chat_history_db_service[ip])
        return JSONResponse(content={"message": db_response})

    history = chat_history_general[ip]["history"]
    temp = init_prompt_web.format(userQuery=q)
    print(temp)
    history.append({"role": "user", "content": temp})

    preset = [
        {
            "role": "system",
            "content": vis_sys_prompt,
        },
        *history,
    ]

    completion = await openai_api.chat_completion(
        model="gpt-4",
        messages=preset,
        temperature=0,
        max_tokens=512,
    )

    ai_response = completion.choices[0].message["content"]
    print(ai_response)
    action = parse_action_line(ai_response)
    print(action)
    history.pop(0)

    if action.startswith("@answer:"):
        return JSONResponse(content={"message": action[8:].strip()})

    elif action.startswith("@error:"):
        return JSONResponse(content={"message": action[7:].strip()})

    elif action.startswith("@todo:") or chat_history_db_service[ip]["active"]:
        db_response = await handle_database_interaction(userId, q, chat_history_db_service[ip])
        return JSONResponse(content={"message": db_response})

    elif action.startswith("@vq:"):
        user_id_vq_cache[userId] = q
        return JSONResponse(content={"message": "@vq"})

    else:
        return JSONResponse(content={"message": ai_response})



from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    content = jsonable_encoder({"detail": str(exc)})
    return JSONResponse(content=content, status_code=400)


""" Local Host/Google Collab"""
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
