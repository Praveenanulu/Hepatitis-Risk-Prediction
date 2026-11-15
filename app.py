import os
import gradio as gr
import joblib
import numpy as np
import uuid
import pygame
import warnings
import matplotlib.pyplot as plt

from gtts import gTTS
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from utils import (
    risk_level,
    generate_explanation,
    preventive_recommendations,
    diet_plan,
    probable_hepatitis_type,
)

warnings.filterwarnings("ignore")

# ======================= CSS (PRO UI) =======================
CSS = """
body {
  background: radial-gradient(circle at top, #0f172a, #020617);
  font-family: 'Inter', sans-serif;
  color: #e5e7eb;
}

.gradio-container {
  max-width: 1400px !important;
  margin: auto;
}

.card {
  background: linear-gradient(145deg, #020617, #020617);
  border-radius: 20px;
  padding: 28px;
  box-shadow: 0 30px 60px rgba(0,0,0,0.6);
}

.risk-badge {
  display: inline-block;
  padding: 6px 16px;
  border-radius: 999px;
  font-weight: 700;
  background: linear-gradient(90deg, #22c55e, #16a34a);
  color: black;
  margin-bottom: 10px;
}

.risk-score {
  font-size: 64px;
  font-weight: 900;
  color: #22c55e;
}

button {
  border-radius: 14px !important;
  font-weight: 700 !important;
}

button:hover {
  transform: scale(1.04);
  transition: 0.2s;
}

img {
  border-radius: 18px;
}
"""

# ================= AUDIO =================
pygame.mixer.init()
VOICE_FILE = None

def speak(text, lang):
    global VOICE_FILE
    stop_voice()
    VOICE_FILE = f"voice_{uuid.uuid4()}.mp3"
    gTTS(text=text, lang=lang).save(VOICE_FILE)
    pygame.mixer.music.load(VOICE_FILE)
    pygame.mixer.music.play()

def stop_voice():
    try:
        pygame.mixer.music.stop()
        return "⛔ Voice stopped"
    except:
        return "⚠️ No voice playing"

# ================= PATHS =================
BASE_DIR = os.path.dirname(__file__)
IMAGE_DIR = os.path.join(BASE_DIR, "images")

LIVER_IMAGE = os.path.join(IMAGE_DIR, "pic 1.jpeg")
DIET_IMAGE  = os.path.join(IMAGE_DIR, "pic 5.jpeg")
HEP_IMAGE   = os.path.join(IMAGE_DIR, "pic 7.webp")

# ================= MODEL =================
model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
rf_model = model["rf_model"]
feature_importance = model["feature_importances"]
lr_coefficients = model["lr_coefficients"]

# ================= LANG =================
LANG_MAP = {
    "English": "en",
    "Hindi": "hi",
    "Marathi": "mr",
    "Telugu": "te",
}

# ================= PDF =================
def generate_pdf(text):
    path = os.path.join(BASE_DIR, "Hepatitis_AI_Report.pdf")
    c = canvas.Canvas(path, pagesize=A4)
    t = c.beginText(40, 800)
    for line in text.split("\n"):
        t.textLine(line)
    c.drawText(t)
    c.save()
    return path

# ================= LOGIN =================
def login(name, age, gender, pid):
    if not name or not pid:
        return gr.update(visible=True), gr.update(visible=False), "❌ Fill all details", {}
    return gr.update(visible=False), gr.update(visible=True), "", {
        "name": name, "age": age, "gender": gender, "id": pid
    }

# ================= CHART =================
def risk_chart(prob):
    fig, ax = plt.subplots()
    ax.barh(["Risk Probability"], [prob * 100], color="#22c55e")
    ax.set_xlim(0, 100)
    ax.set_facecolor("#020617")
    fig.patch.set_facecolor("#020617")
    ax.tick_params(colors="white")
    ax.set_title("Hepatitis Risk (%)", color="white")
    return fig

# ================= LAB TESTS =================
def lab_tests(level):
    if level == "High":
        return "🔬 LFT, HBsAg, Anti-HCV, Ultrasound\n🚨 Urgent doctor visit"
    elif level == "Moderate":
        return "🔬 LFT, HBsAg / Anti-HCV\n🩺 Follow-up advised"
    else:
        return "🔬 Routine LFT\n✅ Annual screening"

# ================= PREDICT =================
def predict(
    age, alcohol, smoking, fatigue, jaundice,
    diabetes, transfusion, vaccination,
    doctor_mode, patient
):
    a = 1 if alcohol != "Never" else 0
    s = int(smoking)
    f = int(fatigue)
    j = int(jaundice)

    X = np.array([[age, a, s, f, j]])
    inputs = [age, a, s, f, j]

    prob = rf_model.predict_proba(X)[0][1]
    level, emoji = risk_level(prob)

    explanation = generate_explanation(inputs, feature_importance, lr_coefficients)
    care = preventive_recommendations(level)
    diet = diet_plan(level)
    htype = probable_hepatitis_type(inputs)
    tests = lab_tests(level)

    if doctor_mode:
        care += "\n\n" + tests

    risk_html = f"""
    <div class="card">
        <div class="risk-badge">{level} Risk</div>
        <div class="risk-score">{prob*100:.1f}%</div>
        <p>AI Predicted Hepatitis Probability</p>
    </div>
    """

    pdf = generate_pdf(
        f"Patient: {patient['name']}\nRisk: {level}\n\n{explanation}\n\n{care}\n\n{diet}"
    )

    return (
        risk_html, explanation, htype,
        tests, care, diet,
        risk_chart(prob), pdf,
        explanation, care
    )

# ================= VOICE =================
def explain_by_voice(exp, care, lang, patient):
    speak(f"Hello {patient['name']}. {exp}. {care}", LANG_MAP[lang])
    return "🔊 Voice playing"

# ================= UI =================
with gr.Blocks(css=CSS, title="Hepatitis Risk Prediction System") as app:

    patient_state = gr.State({})
    exp_state = gr.State("")
    care_state = gr.State("")

    login_col = gr.Column(visible=True)
    main_col = gr.Column(visible=False)

    # -------- SCREEN 1: LOGIN --------
    with login_col:
        gr.Markdown("## 🏥 Patient Login")
        name = gr.Textbox(label="Patient Name")
        age_l = gr.Slider(1, 100, label="Age")
        gender = gr.Dropdown(["Male", "Female", "Other"])
        pid = gr.Textbox(label="Patient ID / Email")
        login_btn = gr.Button("Continue")
        login_msg = gr.Markdown()

    # -------- MAIN DASHBOARD --------
    with main_col:
        gr.Markdown("## 🧠 Hepatitis Risk Prediction Dashboard")

        with gr.Row():
            with gr.Column(scale=1):
                age = gr.Slider(1,100,30,label="Age")
                alcohol = gr.Radio(["Never","Occasional","Frequent"], label="Alcohol")
                smoking = gr.Checkbox(label="Smoking")
                fatigue = gr.Checkbox(label="Fatigue")
                jaundice = gr.Checkbox(label="Jaundice")
                diabetes = gr.Checkbox(label="Diabetes")
                transfusion = gr.Checkbox(label="Blood Transfusion")
                vaccination = gr.Radio(
                    ["Fully Vaccinated","Partially Vaccinated","Not Vaccinated"],
                    label="Vaccination Status"
                )
                doctor_mode = gr.Checkbox(label="🩺 Doctor Mode")
                language = gr.Dropdown(list(LANG_MAP.keys()), value="English")
                predict_btn = gr.Button("🚀 Predict Risk")

            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.Tab("📊 Risk Summary"):
                        risk_out = gr.HTML()
                        chart_out = gr.Plot()

                    with gr.Tab("🧠 Explainable AI"):
                        xai_out = gr.Markdown()
                        gr.Image(value=LIVER_IMAGE)

                    with gr.Tab("🧬 Hepatitis Type"):
                        type_out = gr.Markdown()
                        gr.Image(value=HEP_IMAGE)

                    with gr.Tab("🧪 Tests"):
                        tests_out = gr.Markdown()

                    with gr.Tab("🛡️ Care"):
                        care_out = gr.Markdown()

                    with gr.Tab("🥗 Diet"):
                        diet_out = gr.Markdown()
                        gr.Image(value=DIET_IMAGE)

                    with gr.Tab("📄 Report & Voice"):
                        voice_btn = gr.Button("🔊 Hear Explanation")
                        stop_btn = gr.Button("⛔ Stop Voice")
                        voice_status = gr.Markdown()
                        pdf_out = gr.File()

    # -------- EVENTS --------
    login_btn.click(login,[name,age_l,gender,pid],[login_col,main_col,login_msg,patient_state])

    predict_btn.click(
        predict,
        [age,alcohol,smoking,fatigue,jaundice,diabetes,transfusion,vaccination,doctor_mode,patient_state],
        [risk_out,xai_out,type_out,tests_out,care_out,diet_out,chart_out,pdf_out,exp_state,care_state]
    )

    voice_btn.click(explain_by_voice,[exp_state,care_state,language,patient_state],voice_status)
    stop_btn.click(stop_voice,outputs=voice_status)

app.launch()

























