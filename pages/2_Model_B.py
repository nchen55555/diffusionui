
import streamlit as st
from transformers import AutoProcessor, AutoModelForCausalLM, BlipProcessor, BlipForConditionalGeneration
import requests
from PIL import Image
import numpy as np
import torch

from diffusers import StableDiffusionControlNetPipeline
from diffusers.utils import load_image

from cv2 import cv2
from PIL import Image
import numpy as np
import os

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import torch
from diffusers import UniPCMultistepScheduler

st.set_page_config(
    page_title="Realistic Vision V1.4 CUDA Git Large Coco",
    page_icon="🌍",
)

st.title('Realistic Vision V1.4 CUDA Git Large Coco')
temp_prompt = ''

def imageurl_process(url):
    processor = AutoProcessor.from_pretrained("microsoft/git-base-coco")
    model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-coco")
    image = Image.open(requests.get(url, stream=True).raw)
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
    temp_prompt = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    st.info(temp_prompt)
    return temp_prompt

def prompt_process(text, url):
  model = "SG161222/Realistic_Vision_V1.4"
  controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
  pipe = StableDiffusionControlNetPipeline.from_pretrained(
      model, controlnet=controlnet, torch_dtype=torch.float16
  )

  pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

  pipe.enable_model_cpu_offload()

  low_threshold = 100
  high_threshold = 200

  starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
  repetitions = 20
  timings =np.zeros((repetitions,1))
  image = load_image(url)
  image = np.array(image)
  image = cv2.Canny(image, low_threshold, high_threshold)
  image = image[:, :, None]
  image = np.concatenate([image, image, image], axis=2)
  canny_image = Image.fromarray(image)
  add = ", best quality, extremely detailed"
  finalprompt = [text + add][:77]
  generator = [torch.Generator(device="cuda").manual_seed(2)]
  output = pipe(
    finalprompt,
    canny_image,
    negative_prompt=["monochrome, lowres, bad anatomy, worst quality, low quality"],
    generator=generator,
    num_inference_steps=20,
  )
  image = output.images[0]
  return st.image(image)

if 'stage' not in st.session_state:
    st.session_state.stage = 0

def set_stage(stage):
    st.session_state.stage = stage

first_form = st.form('image_url')
url = first_form.text_input('Enter Image URL: ', '')
first_form.form_submit_button('Generate Prompt', on_click=set_stage, args=(1,))

if st.session_state.stage > 0:
    temp_prompt = imageurl_process(url)
    second_form = st.form('prompt')
    topic_text = second_form.text_input('Enter Prompt: ')
    second_form.form_submit_button('Generate Image', on_click=set_stage, args=(2,))
    if st.session_state.stage > 1:
        st.image(prompt_process(topic_text, url))
st.button('Reset', on_click=set_stage, args=(0,))
