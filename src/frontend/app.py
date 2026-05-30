from __future__ import annotations

import os

import requests
import streamlit as st


API_URL = os.getenv("API_URL", "http://backend:8000")


st.set_page_config(page_title="BERT AI Detector", layout="wide")
st.title("BERT AI Detector")

with st.sidebar:
    st.header("Model")
    detector_type = st.selectbox("Scoring mode", ["ensemble", "hf"])
    default_path = (
        "trained_models/ensemble_models_2_3000/manifest.json"
        if detector_type == "ensemble"
        else "trained_models/Models/epoch_03"
    )
    model_path = st.text_input("Model path", value=default_path)
    threshold = st.slider("AI threshold", 0.0, 1.0, 0.5, 0.01)
    max_length = st.number_input("Max length", min_value=16, max_value=4096, value=256, step=16)
    batch_size = st.number_input("Batch size", min_value=1, max_value=256, value=32, step=1)
    device = st.selectbox("Device", ["auto", "cpu", "cuda"])

score_tab, obfuscate_tab = st.tabs(["Score", "Obfuscate"])


def model_payload() -> dict:
    return {
        "detector_type": detector_type,
        "model_path": model_path,
        "threshold": threshold,
        "max_length": int(max_length),
        "batch_size": int(batch_size),
        "device": None if device == "auto" else device,
    }


with score_tab:
    text = st.text_area("Text for scoring", height=260)
    score_clicked = st.button("Score", type="primary", disabled=not text.strip())

    if score_clicked:
        payload = {"texts": [text], **model_payload()}

        try:
            response = requests.post(f"{API_URL}/score", json=payload, timeout=600)
            response.raise_for_status()
            result = response.json()["results"][0]
        except requests.RequestException as exc:
            st.error(f"Backend request failed: {exc}")
        else:
            probability = float(result["probability_ai"])
            label = result["label"]
            left, right = st.columns([1, 2])
            with left:
                st.metric("Prediction", label)
                st.metric("P(AI)", f"{probability:.4f}")
            with right:
                st.progress(min(max(probability, 0.0), 1.0))
                if result.get("best_member"):
                    st.subheader("Best ensemble member")
                    st.json(result["best_member"])
                if result.get("member_probabilities"):
                    st.subheader("Member probabilities")
                    st.bar_chart(result["member_probabilities"])

with obfuscate_tab:
    obfuscation_threshold = st.slider("Rewrite threshold", 0.0, 1.0, 0.8, 0.01)
    obfuscation_text = st.text_area("Text for obfuscation", height=260)
    obfuscate_clicked = st.button("Obfuscate", type="primary", disabled=not obfuscation_text.strip())

    if obfuscate_clicked:
        payload = {
            "texts": [obfuscation_text],
            **model_payload(),
            "sent_threshold": obfuscation_threshold,
            "sent_max_retries": 3,
            "neighbors": 1,
        }

        try:
            response = requests.post(f"{API_URL}/obfuscate", json=payload, timeout=600)
            response.raise_for_status()
            result = response.json()["results"][0]
        except requests.RequestException as exc:
            st.error(f"Backend request failed: {exc}")
        else:
            st.text_area("Obfuscated text", value=result["obfuscated_text"], height=260)
            st.metric("Rewrites", len(result.get("rewrites", [])))
            if result.get("sentence_scores"):
                st.subheader("Sentence scores")
                st.bar_chart(result["sentence_scores"])
            if result.get("rewrites"):
                st.subheader("Rewrite log")
                st.json(result["rewrites"])
