"""
This file contains a wrapper for Huggingface models, implementing various methods used in downstream tasks.
It includes the HuggingfaceModel class that extends the functionality of the WhiteBoxModelBase class.
"""

import sys
from .model_base import WhiteBoxModelBase
import warnings
from PIL import Image
import requests
from transformers import AutoProcessor, LlavaForConditionalGeneration # llava
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration # instructblip
from transformers import Blip2Processor, Blip2ForConditionalGeneration # blip2
from transformers import AutoTokenizer, AutoModel # internVL
from transformers import AutoModelForCausalLM # Qwen 
from transformers.generation import GenerationConfig
from transformers import IdeficsForVisionText2Text # Idefics
from transformers import LlamaTokenizer # cogvlm

import functools
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from fastchat.conversation import get_conv_template
from typing import Optional, Dict, List, Any
import logging
from .huggingface_model_template import LLAVA


class HuggingfaceModel(WhiteBoxModelBase):
    """
    HuggingfaceModel is a wrapper for Huggingface's transformers models.
    It extends the WhiteBoxModelBase class and provides additional functionality specifically
    for handling conversation generation tasks with various models.
    This class supports custom conversation templates and formatting,
    and offers configurable options for generation.
    """

    def __init__(
        self,
        model: Any,
        processor: Any,
        model_name: str,
        generation_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initializes the HuggingfaceModel with a specified model, processor, and generation configuration.

        :param Any model: A huggingface model.
        :param Any processor: A huggingface processor.
        :param str model_name: The name of the model being used. Refer to
            `FastChat conversation.py <https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py>`_
            for possible options and templates.
        :param Optional[Dict[str, Any]] generation_config: A dictionary containing configuration settings for text generation.
            If None, a default configuration is used.
        """
        super().__init__(model, processor)
        self.model_name = model_name
        self.format_str = self.create_format_str()

        if generation_config is None:
            generation_config = {}
        self.generation_config = generation_config

    def create_format_str(self):

        format_str = None
        if "llava" in self.model_name:
            format_str = LLAVA
        elif "instructblip" in self.model_name:
            format_str = ""
        elif "blip2" in self.model_name:
            format_str = ""
        else:
            KeyError(f"The format supporting {self.model_name} is not available jet.")
        return format_str

    def create_conversation_prompt(self, messages, images, clear_old_history=True):
        r"""
        Constructs a conversation prompt that includes the conversation history.

        :param list[str] messages: A list of messages that form the conversation history.
        :param list[str] images: A list of images that form the conversation history.
        :param bool clear_old_history: If True, clears the previous conversation history before adding new messages.
        :return: A string representing the conversation prompt including the history.
        """
        if clear_old_history:
            self.conversation = []
        if isinstance(messages, str):
            messages = [messages]
            images = [images]
        for index, (message, image) in enumerate(zip(messages, images)):
            if "Qwen" in self.model_name:
                text = message
                image = image.replace("data_contrast/images/","https://osbenchtest.s3.us-west-1.amazonaws.com/images/")
                inputs= [{'image': image}, {'text': text}]
            else:
                if "llava" in self.model_name:
                    text = self.format_str.replace('<prompt>', message)
                    if "https:" in image:
                        image = Image.open(requests.get(image, stream=True).raw)
                    else:
                        image = Image.open(image)
                    inputs = self.processor(text=text, images=image, return_tensors="pt").to(self.device)
                elif "instructblip" in self.model_name:
                    text = message
                    image = Image.open(image).convert("RGB")
                    inputs = self.processor(text=text, images=image, return_tensors="pt").to(self.device)
                elif "blip2" in self.model_name:
                    text = message
                    image = Image.open(image).convert('RGB')
                    inputs = self.processor(text=text, images=image, return_tensors="pt").to(self.device)
                elif "idefics" in self.model_name:
                    text = message
                    image = image.replace("data_contrast/images/","https://osbenchtest.s3.us-west-1.amazonaws.com/images/")
                    inputs = self.processor([image, image], return_tensors="pt").to(self.device)
                elif "cogvlm" in self.model_name:
                    text = message
                    image = Image.open(image).convert('RGB')
                    inputs = self.model.build_conversation_input_ids(self.processor, query=text, history=[], images=[image])  # chat mode
                    inputs = {
                        'input_ids': inputs['input_ids'].unsqueeze(0).to(self.device),
                        'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to(self.device),
                        'attention_mask': inputs['attention_mask'].unsqueeze(0).to(self.device),
                        'images': [[inputs['images'][0].to(self.device).to(torch.bfloat16)]],
                    }
                elif "Yi-VL" in self.model_name:
                    # torch==2.0.1 transformers>=4.36.2
                    text = message
                    image = Image.open(image).convert("RGB")
                    inputs = self.processor(text=text, images=image, return_tensors="pt").to(self.device)
                elif "MiniCPM" in self.model_name or "OmniLMM-12B" in self.model_name:
                    # torch==2.1.2 torchvision==0.16.2 transformers==4.40.0 https://github.com/OpenBMB/MiniCPM-V/blob/main/requirements.txt
                    inputs = ""
                elif "internlm" in self.model_name:
                    text = "<ImageHere>" + message
                    inputs = [text, image]
                elif "InternVL" in self.model_name:
                    text = message
                    image = load_image(image, max_num=6).to(torch.bfloat16).to(self.device)
                    inputs = [text, image]
            self.conversation.append(inputs)
        return self.conversation

    def generate(self, messages, images, input_field_name='input_ids', **kwargs):
        r"""
        Generates a response for the given messages within a single conversation.

        :param list[str]|str messages: The text input by the user. Can be a list of messages or a single message.
        :param list[str]|str images: The image input by the user. Can be a list of images or a single image.
        :param str input_field_name: The parameter name for the input message in the model's generation function.
        :param dict kwargs: Optional parameters for the model's generation function, such as 'temperature' and 'top_p'.
        :return: A string representing the pure response from the model, containing only the text of the response.
        """
        if isinstance(messages, str):
            messages = [messages]
            images = [images]
        prompts = self.create_conversation_prompt(messages, images)

        # Generate
        for prompt in prompts:
            with torch.no_grad():
                if "llava" in self.model_name:
                    generate_ids = self.model.generate(
                        **prompt,                     
                        do_sample=False,
                        num_beams=5,
                        max_length=1000,
                        min_length=10,
                        top_p=1,
                        repetition_penalty=1.5,
                        length_penalty=1.0,
                        temperature=0)
                    output = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

                    # Find the index of "ASSISTANT:"
                    index = output.find("ASSISTANT:")

                    # Extract the substring after "ASSISTANT:"
                    assistant_response = output[index + len("ASSISTANT:"):].strip()

                elif "instructblip" in self.model_name:
                    generate_ids = self.model.generate(
                        **prompt,
                        do_sample=False,
                        num_beams=5,
                        max_length=1000,
                        min_length=10,
                        top_p=1,
                        repetition_penalty=1.5,
                        length_penalty=1.0,
                        temperature=0,
                    )
                    output = self.processor.batch_decode(generate_ids, skip_special_tokens=True)[0]

                    assistant_response = output.strip()

                elif "blip2" in self.model_name:
                    generate_ids = self.model.generate(
                        **prompt,
                        do_sample=False,
                        num_beams=5,
                        max_length=1000,
                        min_length=10,
                        top_p=1,
                        repetition_penalty=1.5,
                        length_penalty=1.0,
                        temperature=0,
                    )
                    output = self.processor.batch_decode(generate_ids, skip_special_tokens=True)[0]

                    assistant_response = output.strip()

                elif "Qwen" in self.model_name:
                    query = self.processor.from_list_format(prompt)

                    response, history = self.model.chat(self.processor, query=query, history=None)

                    assistant_response = response

                elif "idefics" in self.model_name:
                    # Generation args
                    bad_words_ids = self.processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

                    generated_ids = self.model.generate(**prompt, 
                                                        bad_words_ids=bad_words_ids,                     
                                                        do_sample=False,
                                                        num_beams=5,
                                                        max_length=1000,
                                                        min_length=10,
                                                        top_p=1,
                                                        repetition_penalty=1.5,
                                                        length_penalty=1.0,
                                                        temperature=0
                                                        )
                    output = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    assistant_response = output.strip()

                elif "cogvlm" in self.model_name:
                    outputs = self.model.generate(**prompt,                                                         
                                                    do_sample=False,
                                                    num_beams=5,
                                                    max_length=1000,
                                                    min_length=10,
                                                    top_p=1,
                                                    repetition_penalty=1.5,
                                                    length_penalty=1.0,
                                                    temperature=0)
                    output = outputs[:, prompt['input_ids'].shape[1]:]
                    assistant_response = self.processor.decode(output[0])
                
                elif "internlm" in self.model_name:
                    query, image = prompt
                    with torch.cuda.amp.autocast():
                        response, _ = self.model.chat(self.processor, 
                                                    query=query, 
                                                    image=image, 
                                                    history=[], 
                                                    do_sample=False,
                                                    num_beams=5,
                                                    max_length=1000,
                                                    min_length=10,
                                                    top_p=1,
                                                    repetition_penalty=1.5,
                                                    length_penalty=1.0,
                                                    temperature=0)

                    assistant_response = response

                elif "InternVL" in self.model_name:

                    query, image = prompt
                    response = self.model.chat(self.processor, image, query,                                                   
                                               do_sample=False,
                                               num_beams=5,
                                               max_length=1000,
                                               min_length=10,
                                               top_p=1,
                                               repetition_penalty=1.5,
                                               length_penalty=1.0,
                                               temperature=0)

        # multi-round conversation for future development
        return assistant_response

    def __call__(self, *args, **kwargs):
        r"""
        Allows the HuggingfaceModel instance to be called like a function, which internally calls the model's
        __call__ method.

        :return: The output from the model's __call__ method.
        """
        return self.model(*args, **kwargs)

    @property
    def device(self):
        return self.model.device

    @property
    def dtype(self):
        return self.model.dtype

    @property
    def bos_token_id(self):
        return self.tokenizer.bos_token_id

    @property
    def eos_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def pad_token_id(self):
        return self.tokenizer.pad_token_id


def from_pretrained(model_name_or_path: str, model_name: str, processor_name_or_path: Optional[str] = None,
                    dtype: Optional[torch.dtype] = None, **generation_config: Dict[str, Any]) -> HuggingfaceModel:
    """
    Imports a Hugging Face model and tokenizer with a single function call.

    :param str model_name_or_path: The identifier or path for the pre-trained model.
    :param str model_name: The name of the model, used for generating conversation template.
    :param Optional[str] processor_name_or_path: The identifier or path for the pre-trained tokenizer.
        Defaults to `model_name_or_path` if not specified separately.
    :param Optional[torch.dtype] dtype: The data type to which the model should be cast.
        Defaults to None.
    :param generation_config: Additional configuration options for model generation.
    :type generation_config: dict

    :return HuggingfaceModel: An instance of the HuggingfaceModel class containing the imported model and tokenizer.

    .. note::
        The model is loaded for evaluation by default. If `dtype` is specified, the model is cast to the specified data type.
        The `tokenizer.padding_side` is set to 'right' if not already specified.
        If the tokenizer has no specified pad token, it is set to the EOS token, and the model's config is updated accordingly.

    **Example**

    .. code-block:: python

        model = from_pretrained("llava-hf/llava-1.5-7b-hf", '"llava-hf/llava-1.5-7b-hf", dtype=torch.float32, max_length=512)
    """
    if dtype is None:
        dtype = 'auto'
    device_map = "auto"
    if generation_config:
        if "device_map" in generation_config["generation_config"]:
            device_map = generation_config["generation_config"]["device_map"]

    device_map = "cuda:4"

    if 'llava' in model_name_or_path:

        model = LlavaForConditionalGeneration.from_pretrained(model_name_or_path, device_map=device_map, trust_remote_code=True, low_cpu_mem_usage=True, torch_dtype=dtype).eval()

        if processor_name_or_path is None:
            processor_name_or_path = model_name_or_path
        processor = AutoProcessor.from_pretrained(processor_name_or_path, trust_remote_code=True)
    
    elif "instructblip" in model_name_or_path:

        model = InstructBlipForConditionalGeneration.from_pretrained(model_name_or_path, device_map=device_map, trust_remote_code=True, low_cpu_mem_usage=True, torch_dtype=dtype).eval()

        if processor_name_or_path is None:
            processor_name_or_path = model_name_or_path
        processor = InstructBlipProcessor.from_pretrained(processor_name_or_path, trust_remote_code=True)

    elif "blip2" in model_name_or_path:

        processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        model = Blip2ForConditionalGeneration.from_pretrained(model_name_or_path, device_map=device_map, trust_remote_code=True, low_cpu_mem_usage=True, torch_dtype=dtype).eval()

        if processor_name_or_path is None:
            processor_name_or_path = model_name_or_path
        processor = Blip2Processor.from_pretrained(processor_name_or_path, trust_remote_code=True)
    
    elif "Qwen" in model_name_or_path:

        processor = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map=device_map, trust_remote_code=True).eval()

    elif "idefics" in model_name_or_path:

        processor = AutoProcessor.from_pretrained("HuggingFaceM4/idefics-9b", trust_remote_code=True)
        model = IdeficsForVisionText2Text.from_pretrained(model_name_or_path, device_map=device_map, trust_remote_code=True).eval()

    elif "cogvlm" in model_name_or_path:

        processor = LlamaTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5', trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map=device_map,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).eval()

    elif "Yi-VL" in model_name_or_path:

        processor = AutoProcessor.from_pretrained("01-ai/Yi-VL-6B")
        model = AutoModelForCausalLM.from_pretrained("01-ai/Yi-VL-6B",
                                                    device_map=device_map,
                                                    low_cpu_mem_usage=True,
                                                    trust_remote_code=True).eval()
    elif "internlm" in model_name_or_path:

        processor = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                    device_map=device_map,
                                                    low_cpu_mem_usage=True,
                                                    torch_dtype=torch.float16,
                                                    trust_remote_code=True).eval()
        
    elif "InternVL" in model_name_or_path:

        processor = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(
                model_name_or_path,
                torch_dtype=torch.bfloat16,
                device_map=device_map,
                low_cpu_mem_usage=True,
                trust_remote_code=True).eval()

    else:
        KeyError(f"The {model_name_or_path} is not available jet.")

    return HuggingfaceModel(model, processor, model_name=model_name, generation_config=generation_config)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=6):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values