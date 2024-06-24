from PIL import Image
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load processor and model
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
# Load model without Accelerate-specific parameters
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")


# Image captioning
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)
generated_ids = model.generate(**inputs)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
print("Image Caption:", generated_text)

# Visual question answering
prompt = "Question: how many cats are there? Answer:"
inputs = processor(images=image, text=prompt, return_tensors="pt").to(device="cuda", dtype=torch.float16)

generated_ids = model.generate(**inputs)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
print("VQA Answer:", generated_text)
