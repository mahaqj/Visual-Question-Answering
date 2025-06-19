import streamlit as st
from PIL import Image
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# load model and processor (cached to avoid reloading every time)
@st.cache_resource
def load_model():
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=dtype)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return processor, model

processor, model = load_model()

st.title("Visual Question Answering")
st.write("Upload an image and ask a question about it!")

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
question = st.text_input("Ask a question about the image")

# once both image and question are provided:
if uploaded_image and question:
    image = Image.open(uploaded_image).convert("RGB")
    prompt = f"Question: {question} Answer:"
    
    # adjust tensor dtype for device
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device, dtype=dtype)

    # generate answer
    out = model.generate(**inputs)
    decoded = processor.tokenizer.decode(out[0], skip_special_tokens=True)
    if "Answer:" in decoded:
        answer = decoded.split("Answer:")[-1].strip()
    else:
        answer = decoded.strip()

    # post-process to detect bad answers
    question_lower = question.lower()
    answer_lower = answer.lower()

    # basic heuristics: too short, answer contains question keyword, or just repeats
    if (len(answer) < 3 or answer_lower in question_lower or answer_lower.strip(".?") == question_lower.strip(".?")):
        answer = "I'm not sure about that."

    # display results
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.markdown(f"**Q:** {question}")
    st.markdown(f"**A:** {answer}")