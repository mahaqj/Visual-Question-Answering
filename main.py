import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image

image_path = "car.jpg"
image = Image.open(image_path).convert('RGB')

# load processor and blip-2 model
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=dtype)
model.to("cuda" if torch.cuda.is_available() else "cpu")

question = "Question: What is the color of the car? Answer:"
inputs = processor(images=image, text=question, return_tensors="pt").to(model.device, dtype=dtype)
out = model.generate(**inputs) # generate answer
answer = processor.tokenizer.decode(out[0], skip_special_tokens=True)

print(f"Q: {question}")
print(f"A: {answer}")