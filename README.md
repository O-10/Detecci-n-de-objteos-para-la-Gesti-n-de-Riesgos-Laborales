Main file path: main.py
Diseño de sistema de vision computarizada que permite la detección automatica de objetos y corelación semantica del riesgo en entornos laborales como elementos de protección personal, maquinaría, comportamientos inseguros.
import streamlit as st
import cv2
import pandas as pd
import os
import tempfile
import random
from datetime import datetime, timedelta
from roboflow import Roboflow
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Inicialización de variables
keywords_db = {
    "casco": "protección cabeza, evitar golpes",
    "arnés": "prevención caídas, altura",
    "chaleco": "alta visibilidad, maquinaria",
    "persona": "presencia humana, riesgo exposición",
    "escalera": "trabajo en altura, caída",
    "andamio": "estructura elevada, colapso"
}

def get_keywords(obj):
    corpus = list(keywords_db.values())
    keys = list(keywords_db.keys())
    if obj in keys:
        vectorizer = TfidfVectorizer()
        tfidf = vectorizer.fit_transform(corpus + [obj])
        sim = cosine_similarity(tfidf[-1], tfidf[:-1])
        idx = sim.argmax()
        return corpus[idx].split(", ")
    else:
        return ["riesgo", "precaución", "evaluar"]

def asignar_valor(lista, rep):
    base = random.choice(lista)
    return min(base + int(rep / 2), max(lista))

# Conexión a Roboflow
rf = Roboflow(api_key="ZTgQTJF0CA75bTfQixhE")
project = rf.workspace().project("construccion-oscar")
model = project.version(1).model

# Interfaz de usuario
st.title("🔍 RiesgIA - Detección Inteligente de Riesgos Laborales")
st.write("Sube un video para analizar riesgos de seguridad laboral con IA.")

video_file = st.file_uploader("📤 Cargar video (.mp4, .avi)", type=["mp4", "avi"])

if video_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".avi") as temp_input:
        temp_input.write(video_file.read())
        video_path = temp_input.name

    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    output_path = os.path.join(tempfile.gettempdir(), "video_deteccion_riesgos.avi")
    fps = 10.0
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    eventos = []
    frame_number = 0
    start_time = datetime.strptime("08:00:00", "%H:%M:%S")

    stframe = st.empty()
    progress = st.progress(0)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_number > 150:
            break

        frame_path = os.path.join(tempfile.gettempdir(), f"frame_{frame_number}.jpg")
        cv2.imwrite(frame_path, frame)

        try:
            result = model.predict(frame_path, confidence=40, overlap=30)
            predictions = result.json().get("predictions", [])
        except:
            predictions = []

        tiempo_simulado = (start_time + timedelta(seconds=frame_number * 2)).strftime("%H:%M:%S")

        for pred in predictions:
            objeto = pred["class"]
            palabras_clave = get_keywords(objeto)
            eventos.append({
                "frame": frame_number,
                "tiempo": tiempo_simulado,
                "objeto": objeto,
                "confidence": pred["confidence"],
                "palabras_clave": ", ".join(palabras_clave)
            })
            x, y, w, h = int(pred["x"]), int(pred["y"]), int(pred["width"]), int(pred["height"])
            cv2.rectangle(frame, (x - w//2, y - h//2), (x + w//2, y + h//2), (0, 255, 0), 2)
            cv2.putText(frame, objeto, (x - w//2, y - h//2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        out.write(frame)
        stframe.image(frame, channels="BGR")
        progress.progress(min(frame_number / total_frames, 1.0))
        frame_number += 1

    cap.release()
    out.release()

    df = pd.DataFrame(eventos)
    df["repeticiones"] = df.groupby(["frame", "objeto"])["objeto"].transform("count")

    deficiencia_vals = [10, 6, 4, 2]
    exposicion_vals = [4, 3, 2, 1]
    consecuencia_vals = [100, 60, 25, 10]

    calculos = []
    for _, row in df.iterrows():
        rep = row["repeticiones"]
        d = asignar_valor(deficiencia_vals, rep)
        e = asignar_valor(exposicion_vals, rep)
        c = asignar_valor(consecuencia_vals, rep)
        p = d * e
        riesgo = p * c
        aceptabilidad = "🟥 No Aceptable" if riesgo >= 600 else "🟧 Aceptable con Control" if riesgo >= 150 else "🟩 Aceptable"
        calculos.append((d, e, p, c, riesgo, aceptabilidad))

    df[["deficiencia", "exposicion", "probabilidad", "consecuencia", "peligrosidad", "aceptabilidad"]] = pd.DataFrame(calculos, index=df.index)
    df["probabilidad_acumulada"] = df["probabilidad"].cumsum() / df["frame"].nunique()
    df["peligrosidad_acumulada"] = df["peligrosidad"].cumsum() / df["frame"].nunique()

    st.success("✅ Procesamiento finalizado")

    st.download_button("📥 Descargar CSV de riesgos", df.to_csv(index=False).encode("utf-8"), "riesgos_detectados.csv", "text/csv")
    st.video(output_path)

    # Mostrar gráfico
    st.subheader("📊 Gráfico de Riesgos Acumulados")
    fig, ax = plt.subplots()
    ax.plot(df["frame"], df["probabilidad_acumulada"], label="Probabilidad Acumulada", color="blue")
    ax.plot(df["frame"], df["peligrosidad_acumulada"], label="Peligrosidad Acumulada", color="red")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Valor")
    ax.set_title("Riesgos Acumulados")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
