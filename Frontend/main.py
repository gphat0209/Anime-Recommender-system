# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import requests

# ---------------- Config ----------------
st.set_page_config(page_title="Anime Classifier", page_icon="🎬", layout="wide")

# ---------------- Session ----------------
if "api_base" not in st.session_state:
    st.session_state.api_base = "http://localhost:8000"

# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown("### 🎬 Anime Manager")
    st.divider()

    nav = st.radio("Điều hướng", ["🔎 Search", "⭐ Evaluate"], label_visibility="collapsed")

    st.divider()
    st.markdown("#### ⚙️ Settings")
    st.text_input("API Base URL", key="api_base")

# ---------------- Helpers ----------------
def call_search(keyword: str, genres: list[str] = None):
    """Gọi API search anime theo từ khóa"""
    url = f"{st.session_state.api_base}/anime/search"
    body = {"q": keyword, "genres": genres}
    try:
        resp = requests.get(url, json=body, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"Lỗi search: {e}")
        return None

def call_evaluate(synopsis: str, genres: list[str] = None):
    """Gọi API đánh giá anime"""
    url = f"{st.session_state.api_base}/anime/evaluate"
    body = {"synopsis": synopsis, "genres": genres}
    try:
        resp = requests.post(url, json=body, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"Lỗi evaluate: {e}")
        return None

# ---------------- Views ----------------
def view_search():
    st.title("🔎 Search Anime")
    st.markdown("<div style='margin-top:0.6rem'></div>", unsafe_allow_html=True)
    keyword = st.text_input(
    "Enter anime description", 
    placeholder="VD: Anime about ninjas, ...",
    )  
    st.markdown("<div style='margin-top:0.6rem'></div>", unsafe_allow_html=True)
    genres = st.multiselect("Chose genres", ["Action", "Romance", "Comedy", "Drama", "Fantasy", "Sci-Fi"])

    run = st.button("Search", type="primary")

    if run and keyword.strip():
        with st.spinner("Finding..."):
            data = call_search(keyword.strip(), genres)
            if not data:
                st.warning("Couldn't find any anime.")
                return

            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True)

def view_evaluate():
    st.title("⭐ Evaluate Anime")
    st.markdown("<div style='margin-top:0.6rem'></div>", unsafe_allow_html=True)
    synopsis = st.text_area("Enter synopsis", placeholder="A girl discovers she has magical powers...")
    st.markdown("<div style='margin-top:0.6rem'></div>", unsafe_allow_html=True)
    genres = st.multiselect("Chose genres", ["Action", "Romance", "Comedy", "Drama", "Fantasy", "Sci-Fi"])
    run = st.button("Evaluate", type="primary")

    if run and synopsis.strip():
        with st.spinner("Evaluating based on past animes..."):
            result = call_evaluate(synopsis, genres)
            if not result:
                st.warning("No result from backend.")
                return
            st.success(f"📊 Prediction: **{result.get('label','N/A')}**")
            st.json(result)

# ---------------- Router ----------------
if __name__ == "__main__" or True:
    if "Search" in nav:
        view_search()
    else:
        view_evaluate()
