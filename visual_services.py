import json
import os
import requests
from PIL import Image
import torch
import numpy as np

from fastapi import HTTPException
from transformers import BlipProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering
from config import OCR_API_KEY

import requests


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