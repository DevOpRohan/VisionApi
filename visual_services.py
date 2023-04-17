import os
import requests
from PIL import Image
import torch
import concurrent.futures
import json
import numpy as np
import os
import ast

from langchain.chat_models import ChatOpenAI
from fastapi import HTTPException
from transformers import BlipProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering, pipeline
from config import OCR_API_KEY

os.environ['OPENAI_API_TOKEN'] = OCR_API_KEY


class Language:
    Arabic = 'ara'
    Bulgarian = 'bul'
    Chinese_Simplified = 'chs'
    Chinese_Traditional = 'cht'
    Croatian = 'hrv'
    Danish = 'dan'
    Dutch = 'dut'
    English = 'eng'
    Finnish = 'fin'
    French = 'fre'
    German = 'ger'
    Greek = 'gre'
    Hungarian = 'hun'
    Korean = 'kor'
    Italian = 'ita'
    Japanese = 'jpn'
    Norwegian = 'nor'
    Polish = 'pol'
    Portuguese = 'por'
    Russian = 'rus'
    Slovenian = 'slv'
    Spanish = 'spa'
    Swedish = 'swe'
    Turkish = 'tur'


# OCR
class Ocr:
    def __init__(
            self,
            endpoint='https://api.ocr.space/parse/image',
            api_key=OCR_API_KEY,
            language=Language.English,
            ocr_engine=5,
            **kwargs,
    ):
        """
        :param endpoint: API endpoint to contact
        :param api_key: API key string
        :param language: document language
        :param **kwargs: other settings to API
        """
        self.endpoint = endpoint
        self.payload = {
            'isOverlayRequired': True,
            'apikey': api_key,
            'language': language,
            'OCREngine': ocr_engine,
            **kwargs
        }

    def _parse(self, raw):
        if type(raw) == str:
            raise Exception(raw)
        if raw['IsErroredOnProcessing']:
            raise Exception(raw['ErrorMessage'][0])
        return raw['ParsedResults'][0]['ParsedText']

    def ocr_file(self, fp):
        """
        Process image from a local path.
        :param fp: A path or pointer to your file
        :return: Result in JSON format
        """
        with (open(fp, 'rb') if type(fp) == str else fp) as f:
            r = requests.post(
                self.endpoint,
                files={'filename': f},
                data=self.payload,
            )
        print(self._parse(r.json()))
        return self._parse(r.json())

    def ocr_url(self, url):
        """
        Process an image at a given URL.
        :param url: Image url
        :return: Result in JSON format.
        """
        data = self.payload
        data['url'] = url
        r = requests.post(
            self.endpoint,
            data=data,
        )
        print(self._parse(r.json()))
        return self._parse(r.json())

    def ocr_base64(self, base64image):
        """
        Process an image given as base64.
        :param base64image: Image represented as Base64
        :return: Result in JSON format.
        """
        data = self.payload
        data['base64Image'] = base64image
        r = requests.post(
            self.endpoint,
            data=data,
        )
        return self._parse(r.json())


# Image captioning
class ImageCaptioning:
    def __init__(self, device):
        self.device = device
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base", torch_dtype=self.torch_dtype).to(self.device)
        self.model.config.max_new_tokens = 128  # Set max_new_tokens

    def inference(self, image_path):
        inputs = self.processor(Image.open(image_path), return_tensors="pt").to(self.device, self.torch_dtype)
        out = self.model.generate(**inputs)
        captions = self.processor.decode(out[0], skip_special_tokens=True)
        return captions


# Visual Question Answering
class VisualQuestionAnswering:
    def __init__(self, device):
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.device = device
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
        self.model = BlipForQuestionAnswering.from_pretrained(
            "Salesforce/blip-vqa-base", torch_dtype=self.torch_dtype).to(self.device)
        self.model.config.max_new_tokens = 128  # Set max_new_tokens

    def inference(self, inputs):
        image_path, question = inputs.split(",")[0], ','.join(inputs.split(',')[1:])
        raw_image = Image.open(image_path).convert('RGB')
        inputs = self.processor(raw_image, question, return_tensors="pt").to(self.device, self.torch_dtype)
        out = self.model.generate(**inputs)
        answer = self.processor.decode(out[0], skip_special_tokens=True)
        return answer


""" Object Detection """


class ObjectDetection:
    def __init__(self, device):
        if 'cuda' in device:
            self.device = 0
        else:
            self.device = -1

        self.model = pipeline(model="facebook/detr-resnet-50", device=self.device)

    def inference(self, image_path, threshold=0.7):
        image = Image.open(image_path)
        outputs = self.model(image, threshold=threshold)
        return outputs


class ZeroShotObjectDetection:
    def __init__(self, device):
        if 'cuda' in device:
            self.device = 0
        else:
            self.device = -1

        self.model = pipeline(model="google/owlvit-base-patch32", task="zero-shot-object-detection", device=self.device)

    def inference(self, image_path, candidate_labels):
        image = Image.open(image_path)
        outputs = self.model(image, candidate_labels=candidate_labels)
        return outputs


""" Images Utility """


# Save and process image
def save_and_process_image(image_path, user_id):
    """
    1. The image is opened using the Python Imaging Library (PIL).
    2. The image is resized to fit within a 512x512 bounding box while maintaining its aspect ratio.
    The new width and height are rounded to the nearest multiple of 64.
    3. The image is converted to the RGB color space if it's not already in that format.
    4. The resized and converted image is saved as a PNG file with a unique filename in the 'image' directory.
    """
    image_filename = os.path.join('image', f"{user_id}.png")
    os.makedirs('image', exist_ok=True)
    img = Image.open(image_path)
    width, height = img.size
    ratio = min(512 / width, 512 / height)
    width_new, height_new = (round(width * ratio), round(height * ratio))
    width_new = int(np.round(width_new / 64.0)) * 64
    height_new = int(np.round(height_new / 64.0)) * 64
    img = img.resize((width_new, height_new))
    img = img.convert('RGB')
    img.save(image_filename, "PNG")
    return image_filename


# Download image
def download_image(image_url, user_id):
    response = requests.get(image_url)
    if response.status_code == 200:
        image_path = os.path.join('image', f"{user_id}.png")
        with open(image_path, 'wb') as f:
            f.write(response.content)
        return image_path
    else:
        raise HTTPException(status_code=400, detail="Image download failed")

# == PROMPT MANAGER ==

from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)


class PromptManager:
    def __init__(self):
        self.templates = self._initialize_templates()

    def _initialize_templates(self):
        templates = {}

        # System role template
        system_role_template = SystemMessagePromptTemplate.from_template(
            "You are vision an AI system to give the response of the below visual query using various tools.\n"
            "Query:\n```\n{query}\n```\n"
            "You have access to the following tools.\n"
            "[\n\n**ZeroShotObjectDetection**\n"
            "Give an array of labels in specified format as input to this to get to know whether these  are present or  not.\n"
            "Format example\n```\n@ZeroShotObjectDetection:[\"chair\", \"table\", \"glass\"]\n```\n],\n"
            "[\n**VisualQuestionAnswering**\n"
            "Ask simple independent visual questions about image  in the below format to get more details.\n"
            "Format Example\n```\n@VisualQuestionAnswering:[<ques1>,<ques2>,<ques3>]\n```\n\n"
            "Rules\nAtmax no. of ques should be {maxVqQues}\n"
            "Question shouldn't be  about getting text/labels.\n]\n\n"
            "Follow the user's instruction carefully and  always respond in proper format."
        )
        templates["system_role"] = system_role_template

        # User's 1st message template
        user_first_message_template = HumanMessagePromptTemplate.from_template(
            "Input from  other tools:\n"
            "ObjectDetection:\n```\n{ObjectDetectionOutput}\n```\n"
            "ImageCaptioning:\n```\n{ImageCaptioningOutput}\n```\n"
            "TextDetected:\n```\n{OcrOutput}\n```\n\n"
            "Now, if information provided by me is enough, then respond with a answer in format\n"
            "@ans:<answer>\nelse,tell me use which one of the two tool, and wait for my response in the specified format.\n"
            "@<toolKeyword>:<input>"
        )
        templates["user_first_message"] = user_first_message_template

        # User's 2nd message template
        user_second_message_template = HumanMessagePromptTemplate.from_template(
            "Output: {IntermdiateOutput}\n"
            "Now,if you want to use  VisualQuestionAnswering, the respond me in proper format else conclude the answer."
        )
        templates["user_second_message"] = user_second_message_template

        # User's 3rd message template
        user_third_message_template = HumanMessagePromptTemplate.from_template(
            "Output: {IntemediateOutput}\n"
            "Now, conclude  the answer"
        )
        templates["user_third_message"] = user_third_message_template

        return templates

    def format_template(self, template_name, **kwargs):
        template = self.templates.get(template_name)
        if template:
            return template.prompt.format_prompt(**kwargs).to_messages()
        else:
            raise ValueError(f"Template '{template_name}' not found in the PromptManager.")


class Chat:
    def __init__(self):
        self.prompt_manager = PromptManager()
        self.conversation = []

    def add_system_message(self, template_name, **kwargs):
        messages = self.prompt_manager.format_template(template_name, **kwargs)
        # After adding system message, the prompt is
        print(messages)
        self.conversation.extend(messages)

    def add_human_message(self, template_name, **kwargs):
        messages = self.prompt_manager.format_template(template_name, **kwargs)
        self.conversation.extend(messages)

    def add_ai_message(self, content):
        ai_message = AIMessage(content=content)
        self.conversation.append(ai_message)

    def get_conversation(self):
        return self.conversation

    def clear_conversation(self):
        self.conversation = []

    def __str__(self):
        return "\n".join([str(message) for message in self.conversation])


class Vision:
    def __init__(self):
        # Initialize the chat
        print("Initializing chat")
        self.chat = Chat()

        # Initialize the GPT-4 model
        self.chat_openai = ChatOpenAI(temperature=0, model="gpt-4")
        self.output = ""
        # Load the visual Foundations models
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Loading models on device:", device)
        self.image_captioning = ImageCaptioning(device=device)
        print("Image captioning model loaded")
        self.visual_question_answering = VisualQuestionAnswering(device=device)
        print("Visual question answering model loaded")
        self.object_detection = ObjectDetection(device=device)
        print("Object detection model loaded")
        self.zeroshot_object_detection = ZeroShotObjectDetection(device=device)
        print("Zero shot object detection model loaded")
        self.ocr = Ocr()
        print("Models loaded")
        self.image = None

    def _process_ai_response(self, response):
        if response.startswith("@ans:"):
            return response[5:].strip(), True
        elif response.startswith("@ZeroShotObjectDetection:"):
            # Convert AI response to a list of strings
            labels = json.loads(response[25:].strip())

            # Call ZeroShotObjectDetection model here
            output = self.zeroshot_object_detection.inference(self.image, labels)
            return output, False

        elif response.startswith("@VisualQuestionAnswering:"):
            # Print the AI response for debugging purposes
            print("AI response:", response[25:].strip())

            try:
                questions = ast.literal_eval(response[25:].strip())
            except (ValueError, SyntaxError):
                print("Invalid format in AI response:", response[25:].strip())
                questions = []

            # Call VisualQuestionAnswering model here
            output = ""
            for i, question in enumerate(questions, start=1):
                answer = self.visual_question_answering.inference(f"{self.image},{question}")
                output += f"{i}. {answer}\n"
                print(output)

            return output, False  # Return the combined output for all the questions and False instead of None
        else:
            return "", False  # Return an empty string and False instead of None

    def get_answer(self, query, image):
        # Set image
        self.image = image
        # Invoke objectDetection, ocr, and imageCaptioning concurrently
        with concurrent.futures.ThreadPoolExecutor() as executor:
            object_detection_future = executor.submit(self.object_detection.inference, self.image)
            image_captioning_future = executor.submit(self.image_captioning.inference, self.image)
            ocr_future = executor.submit(self.ocr.ocr_file, self.image)

            object_detection_output = object_detection_future.result()
            image_captioning_output = image_captioning_future.result()
            ocr_output = ocr_future.result()

        # Initialize chat by adding system role and user's first message
        self.chat.add_system_message("system_role", maxVqQues=3, query=query)
        self.chat.add_human_message("user_first_message",
                                    ObjectDetectionOutput=object_detection_output,
                                    ImageCaptioningOutput=image_captioning_output,
                                    OcrOutput=ocr_output)

        # Get AI response and process it
        ai_response = self.chat_openai(self.chat.get_conversation())
        self.chat.add_ai_message(ai_response.content)
        self.output, is_final = self._process_ai_response(ai_response.content)

        if not is_final:
            # Add user's 2nd message and get AI response
            self.chat.add_human_message("user_second_message", IntermdiateOutput=self.output)
            ai_response = self.chat_openai(self.chat.get_conversation())
            self.chat.add_ai_message(ai_response.content)
            self.output, is_final = self._process_ai_response(ai_response.content)

            if not is_final:
                # Add user's 3rd message and get AI response
                self.chat.add_human_message("user_third_message", IntemediateOutput=self.output)
                ai_response = self.chat_openai(self.chat.get_conversation())
                self.chat.add_ai_message(ai_response.content)
                self.output, _ = self._process_ai_response(ai_response.content)

        # Clear the chat and return the final answer
        self.chat.clear_conversation()
        return self.output


# print("hello")
# Vis = Vision()
# print("hi")
#
# ans = Vis.get_answer(
#     "If there is book, what's written on that also check whether there is a person or not also tell me the colour of table and book?",
#     "3.png")
#
# print(ans)