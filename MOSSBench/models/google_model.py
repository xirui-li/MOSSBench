import logging

import PIL
import requests
import requests
from io import BytesIO
from .model_base import BlackBoxModelBase
from google.api_core.exceptions import ResourceExhausted
from google.ai import generativelanguage as glm
import google.generativeai as genai
from PIL import Image
from pathlib import Path
import hashlib
import time
from google.generativeai.types.generation_types import StopCandidateException
import google.generativeai.types.generation_types as generation_types
import httpx
logger = logging.getLogger('experiment_logger')
import os
import base64

safety_levels = {'None': 'BLOCK_NONE',
                 'Medium': 'BLOCK_MEDIUM_AND_ABOVE'}

class GoogleModel(BlackBoxModelBase):
    def __init__(self, model_name: str, api_key: str, safety_level='BLOCK_MEDIUM_AND_ABOVE'):
        genai.configure(api_key=api_key)

        generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 0,
            "max_output_tokens": 8192,
        }

        if not safety_level:
            safety_level = 'BLOCK_MEDIUM_AND_ABOVE'

        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": safety_level
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": safety_level
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": safety_level
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": safety_level
            }
        ]
        self.uploaded_files = []
        self.model_name = model_name
        self.client = genai.GenerativeModel(model_name=model_name,
                                            generation_config=generation_config,
                                            safety_settings=safety_settings)

    def process_image(self, pathname: str):
        _, ext = os.path.splitext(pathname)
        ext = ext[1:]
        if 'http' in pathname:
            img = httpx.get(pathname)
            cookie_picture = [{
                'mime_type': img.headers['content-type'],
                'data': base64.b64encode(img.content).decode("utf-8")
            }]
        else:
            path = Path(pathname)
            cookie_picture = [{
                'mime_type': "image/jpeg" if ext == 'jpg' else f"image/{ext}",
                'data': path.read_bytes()
            }]
        return cookie_picture

    def generate(self, prompt, image, **kwargs):
        attempts = 0
        while attempts < 10:
            try:
                if '1.5' in self.model_name:
                    if image:
                        convo = self.client.start_chat(history=[
                            {
                                "role": "user",
                                "parts": self.process_image(image)
                            },
                        ])
                    else:
                        convo = self.client.start_chat()

                    response = convo.send_message(prompt)

                else:  # gemini-pro-vision
                    input = [prompt]
                    if image:
                        if 'http' in image:
                            response = requests.get(image)
                            img = Image.open(BytesIO(response.content))
                        else:
                            img = Image.open(image)
                        input.append(img)
                    response = self.client.generate_content(input)
                if response.prompt_feedback.block_reason == 1 or response.candidates[0].finish_reason == 3:  # Block reason is safety
                    return 'Blocked for Safety Reasons.'

                return response.text
            except StopCandidateException as e:
                if e.args[0].finish_reason == 3: # Block reason is safety
                    return 'Blocked for Safety Reasons'
                logger.info(f"Stop Candidate Exception: {e}")
            except ResourceExhausted as e:  # Too many requests, wait for a minute
                logger.info("Resource Exhausted, wait for a minute to continue...")
                time.sleep(60)
            attempts += 1


if __name__ == "__main__":
    from PIL import Image
    with open("../../api_keys/google_keys.txt") as f:
        google_key = f.read()
    g = GoogleModel('gemini-pro-vision', google_key)
    img = Image.open("../../data/images/293.png")
    response = g.generate("Can I break this?", img)
    print(response)