# Visual Question Answering (VQA) App

This is a Visual Question Answering web app using BLIP-2 and Streamlit.  
It lets users upload an image, ask a question (by voice or text), and get a generated answer — with optional voice output.

---

## Features

- Upload an image (`.jpg`, `.jpeg`, `.png`)
- Ask a question via **text** or **voice input**
- AI answers your question using **BLIP-2**
- Optional voice output using `gTTS` + `playsound`

---

## Setup Instructions

### 1. Clone or Download the Project

```bash
git clone https://github.com/mahaqj/Visual-Question-Answering.git
```

### 2. (Optional) Create a Virtual Environment

```bash
python -m venv vqaenv
vqaenv\Scripts\activate  # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the App

```bash
streamlit run app.py
```

---

## Example Workflow

1. Upload an image: `car2.png`
2. Ask: “What colour is the car?”
3. Receive answer like: “blue”
4. Click “Speak Answer” to hear it spoken aloud

---

## Voice Input

- Voice input works **in-browser** using `streamlit-audiorecorder`
- Transcription is done using **Google Speech Recognition**
*Note: It is not currently working correctly

---

## Voice Output (Optional)

- Uses `gTTS` to generate speech and `playsound` to play it
- If output isn't audible:
  - Check your volume and output device
  - Try opening `output.mp3` manually
  - Use `os.system("start output.mp3")` instead of `playsound` if needed

---

## Example Result

![Screenshot 2025-06-19 135754](https://github.com/user-attachments/assets/021e1b04-7af0-42b0-b46c-91054eef8e92)
![Screenshot 2025-06-19 135815](https://github.com/user-attachments/assets/72eb590a-6d68-494a-bfb9-5e28613e959e)
