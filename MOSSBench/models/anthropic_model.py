import os
import io
import cv2
import httpx
import base64
import logging
import warnings
from PIL import Image
from .model_base import BlackBoxModelBase
from anthropic import Anthropic
from fastchat.conversation import get_conv_template

class AnthropicModel(BlackBoxModelBase):
    def __init__(self, model_name: str, api_keys: str, generation_config=None):
        """
        Initializes the OpenAI model with necessary parameters.
        :param str model_name: The name of the model to use. (claude-3-opus-20240229, claude-3-sonnet-20240229, claude-3-haiku-20240307)
        :param str api_keys: API keys for accessing the OpenAI service.
        :param str template_name: The name of the conversation template, defaults to 'chatgpt'.
        :param dict generation_config: Configuration settings for generation, defaults to an empty dictionary.
        """
        self.client = Anthropic(api_key=api_keys)
        self.model_name = model_name
        self.conversation = get_conv_template('claude')
        self.generation_config = generation_config if generation_config is not None else {}
        self.system_promt = ""
        self.seed = 48

    def set_system_message(self, system_message: str):
        """
        Sets a system message for the conversation.
        :param str system_message: The system message to set.
        """
        self.conversation.system_message = system_message

    def generate(self, messages, images, budget=5, clear_old_history=True, **kwargs):
        """
        Generates a response based on messages that include conversation history.
        :param list[str]|str messages: A list of messages or a single message string.
        :param list[str]|str messages: A list of images or a single image path.
        :param bool clear_old_history: If True, clears the old conversation history before adding new messages.
        :return str: The response generated by the OpenAI model based on the conversation history.
        """
        if clear_old_history:
            self.input_list = []
        if isinstance(messages, str):
            messages = [messages]
            images = [images]

        scale = budget
        while budget > 0:
            try:

                self.inputs = []
                for index, (message, image) in enumerate(zip(messages, images)):

                    self.input = {}
                    self.input['role'] = 'user'
                    self.input['content'] = []

                    text_conv = {"type": "text", "text": message}
                    self.input['content'].append(text_conv)

                    if "http" in image:
                        img = httpx.get(image)
                        image_data = base64.b64encode(img.content).decode("utf-8")
                        image_conv = {"type": "image", "source": {"type": "base64",
                                "media_type": img.headers['content-type'],
                                "data": image_data}}
                    else:
                        base64_image = self.encode_image(image, scale=scale-budget)
                        _, ext = os.path.splitext(image)
                        ext = ext[1:]
                        image_conv = {"type": "image", "source": {"type": "base64",
                                "media_type": "image/jpeg" if ext == 'jpg' else f"image/{ext}",
                                "data": base64_image}}
                    self.input['content'].append(image_conv)
                    self.inputs.append(self.input)

                    # if "https:" in image:
                    #     image_data = base64.b64encode(httpx.get(image).content).decode("utf-8")
                    #     image_conv = {"type": "image", "source": {"type": "base64",
                    #             "media_type": "image/png",
                    #             "data": image_data}}
                    # else:
                    #     base64_image = self.encode_image(image)
                    #     image_conv = {"type": "image", "source": {"type": "base64",
                    #             "media_type": "image/png",
                    #             "data": base64_image}}
                    # self.input['content'].append(image_conv)
                    # self.inputs.append(self.input)

                response = self.client.messages.create(
                    model=self.model_name,
                    # msystem=self.system_prompt,
                    messages=self.inputs,
                    max_tokens=1024,
                    temperature=0,
                    **kwargs,
                    **self.generation_config
                )
                return response.content[0].text
            
            except Exception as e:
                print(f"Claude API Error {e}")
                print(f"Try it again with remaining budget {budget}")
                budget -= 1
        

    def batch_generate(self, conversations, batches, **kwargs):
        """
        Generates responses for multiple conversations in a batch.
        :param list[list[str]]|list[str] conversations: A list of conversations, each as a list of messages.
        :param list[list[str]]|list[str] batches: A list of batches, each as a list of images.
        :return list[str]: A list of responses for each conversation.
        """
        responses = []
        for conversation, image in zip(conversations, batches):
            if isinstance(conversation, str):
                warnings.warn('For batch generation based on several conversations, provide a list[str] for each conversation. '
                              'Using list[list[str]] will avoid this warning.')
            responses.append(self.generate(conversation, image, **kwargs))
        return responses
    
    def encode_image(self, image_path, scale=0):
        """
        Encodes an image to base64, resizing it with scale.
        """
        with Image.open(image_path) as img:
            # Adding a loop to check the image size continuously
            scale_factor = 0.9 ** scale
            new_width = int(img.width * scale_factor)
            new_height = int(img.height * scale_factor)
            img = img.resize((new_width, new_height), Image.LANCZOS)
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            print("Adjusted image size: " + str(img_byte_arr.tell()))
            
            img_byte_arr.seek(0)
            return base64.b64encode(img_byte_arr.read()).decode('utf-8')