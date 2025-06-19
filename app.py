import streamlit as st
from PIL import Image
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from gtts import gTTS
from playsound import playsound
import os
from audiorecorder import audiorecorder
import tempfile
import speech_recognition as sr

# voice output using gtts + playsound
def speak_text(text):
    filename = "output.mp3"
    tts = gTTS(text)
    tts.save(filename)
    playsound(filename)
    os.remove(filename)

# load blip-2 processor & model
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

uploaded_image = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

# question input (voice or text)
question = ""
use_voice = st.checkbox("🎤 Use voice input")

if use_voice:
    audio = audiorecorder("🎙️ Start Recording", "⏹️ Stop Recording")

    if len(audio) > 0:
        st.audio(audio, format="audio/wav")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(audio)
            audio_path = f.name

        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
            try:
                question = recognizer.recognize_google(audio_data)
                st.success(f"You said: {question}")
            except sr.UnknownValueError:
                st.error("Could not understand the audio")
            except sr.RequestError:
                st.error("Could not reach Google Speech API")
else:
    question = st.text_input("Ask a question about the image")

# process image + question
if uploaded_image and question:
    image = Image.open(uploaded_image).convert("RGB")
    prompt = f"Question: {question} Answer:"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device, dtype=dtype)
    out = model.generate(**inputs)
    decoded = processor.tokenizer.decode(out[0], skip_special_tokens=True)

    if "Answer:" in decoded:
        answer = decoded.split("Answer:")[-1].strip()
    else:
        answer = decoded.strip()

    question_lower = question.lower()
    answer_lower = answer.lower()

    if (len(answer) < 3 or answer_lower in question_lower or answer_lower.strip(".?") == question_lower.strip(".?")):
        answer = "I'm not sure about that."

    # show results
    st.image(image, caption="Uploaded Image", use_container_width=True)
    st.markdown(f"**Q:** {question}")
    if st.button("🔊 Speak Answer"):
        speak_text(answer)
    st.markdown(f"**A:** {answer}")
