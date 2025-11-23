# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import requests
import os

# ---------------- Config ----------------
st.set_page_config(page_title="Anime Classifier", page_icon="🎬", layout="wide")

# ---------------- Session ----------------
if "api_base" not in st.session_state:
    st.session_state.api_base = os.getenv("API_BASE")

# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown("### 🎬 Anime Manager")
    st.divider()

    nav = st.radio("Điều hướng", ["🔎 Search", "⭐ Evaluate"], label_visibility="collapsed")

    st.divider()
    st.markdown("#### ⚙️ Settings")
    st.text_input("API Base URL", key="api_base")

# ---------------- Helpers ----------------
def call_search(keyword: str):
    """Gọi API search anime theo từ khóa"""
    url = f"{st.session_state.api_base}/anime/search"
    body = {"q": keyword}
    try:
        resp = requests.get(url, params=body, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"Lỗi search: {e}")
        return None


def view_search():
    st.title("🔎 Search Anime")
    st.markdown("<div style='margin-top:0.6rem'></div>", unsafe_allow_html=True)

    keyword = st.text_input(
        "Enter anime description", 
        placeholder="VD: Anime about ninjas, ...",
    )  

    run = st.button("Search", type="primary")

    # 🔹 Khi bấm Search -> lưu vào session_state
    if run and keyword.strip():
        with st.spinner("Finding..."):
            data = call_search(keyword.strip())
            if not data:
                st.warning("Couldn't find any anime.")
                return
            st.session_state["search_results"] = pd.DataFrame(data)

    if "search_results" in st.session_state:
        df = st.session_state["search_results"].copy()
        # df["Link"] = "https://myanimelist.net/anime/" + df["ID"].astype(str)

        # Lấy tất cả genres từ list
        all_genres = sorted({g for row in df["Genres"].dropna() for g in row})
        selected_genres = st.multiselect("🎭 Filter by Genres", options=all_genres)

        if selected_genres:
            mask = df["Genres"].apply(lambda genres: all(g in genres for g in selected_genres))
            df = df[mask]

        # Hiển thị
        df_display = df.copy()
        df_display["Link"] = "https://myanimelist.net/anime/" + df_display["ID"].astype(str)
        display_cols = ["Name", "Type", "Aired", "Episode", "Genres", "Score", "Link"]
        # st.dataframe(df_display[display_cols], use_container_width=True)
        # st.markdown(df_display[display_cols].to_html(escape=False, index=False), unsafe_allow_html=True)
        st.data_editor(
            df_display[display_cols],
            use_container_width=True,
            column_config={
                "Link": st.column_config.LinkColumn("Link", display_text="🔗 Link")
            },
            hide_index=True,
        )

# @st.cache_data
# def fetch_genres():
#     url = f"{st.session_state.api_base}/anime/genres"
#     try:
#         resp = requests.get(url, timeout=30)
#         resp.raise_for_status()
#         return resp.json().get("genres", [])
#     except Exception as e:
#         st.error(f"Lỗi khi lấy danh sách thể loại: {e}")
#         return []

@st.cache_data
def fetch_genres_cached(url):
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json().get("genres", [])

def fetch_genres():
    url = f"{st.session_state.api_base}/anime/genres"
    try:
        return fetch_genres_cached(url)
    except Exception as e:
        st.error(f"Lỗi khi lấy danh sách thể loại: {e}")
        # Không cache lỗi, trả về giá trị tạm thời
        return []

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

def view_evaluate():
    st.title("⭐ Evaluate Anime")
    st.markdown("<div style='margin-top:0.6rem'></div>", unsafe_allow_html=True)
    synopsis = st.text_area("Enter synopsis", placeholder="A girl discovers she has magical powers...")
    st.markdown("<div style='margin-top:0.6rem'></div>", unsafe_allow_html=True)
    all_genres = fetch_genres()
    genres = st.multiselect("Choose genres", all_genres)
    # genres = st.multiselect("Choose genres", ["Action", "Romance", "Comedy", "Drama", "Fantasy", "Sci-Fi"])
    run = st.button("Evaluate", type="primary")

    if run and synopsis.strip():
        with st.spinner("Evaluating based on past animes..."):
            result = call_evaluate(synopsis, genres)
            if not result:
                st.warning("No result from backend.")
                return

            # Hiển thị kết quả dự đoán và phản hồi tự nhiên
            # st.json(result)
            st.markdown(f"### 🎯 Prediction: **{result.get('label','N/A')}**")
            # st.markdown(f"**Confidence:** {result.get('confidence', 0):.2f}")
            st.divider()
            st.markdown("### 💬 Comment")
            st.markdown(result.get("comment", "No comment generated."))
            # print(result.get("comment"))
            # st.write(result.get("comment", "No comment generated."))

# ---------------- Router ----------------
if __name__ == "__main__" or True:
    if "Search" in nav:
        view_search()
    else:
        view_evaluate()
