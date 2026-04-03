import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
import json

# 1. API 설정 (Secrets 활용)
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model = genai.GenerativeModel('gemini-1.5-flash')
else:
    st.error("⚠️ Streamlit Settings > Secrets에 GEMINI_API_KEY를 설정해주세요.")
    st.stop()

st.set_page_config(page_title="TrueWindow Neuro-AI", layout="wide")
st.title("🧠 TrueWindow Neuro-AI Analyzer")

uploaded_file = st.file_uploader("MUSE 2 CSV 파일을 업로드하세요", type=["csv"])

if uploaded_file:
    try:
        # 2. 인코딩 대응 데이터 로드
        try:
            df = pd.read_csv(uploaded_file, encoding='cp949', skiprows=1)
        except:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding='utf-8', skiprows=1)

        targets = ['Alpha_TP9', 'Alpha_AF7', 'Alpha_AF8', 'Alpha_TP10']
        for col in targets:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=['Alpha_TP9']).reset_index(drop=True)

        # 3. AI 분석 구간 탐색
        with st.spinner("AI가 최적 구간을 분석 중입니다..."):
            trend = df[targets].mean(axis=1).fillna(0).tolist()
            step = max(1, len(trend) // 100)
            summary = trend[::step]
            
            prompt = f"EEG: {summary}. Find baseline(0-30) and peak(40-100). Return ONLY JSON: {{\"pre\": 10, \"post\": 80}}"
            response = model.generate_content(prompt)
            res = json.loads(response.text.replace('```json', '').replace('```', '').strip())
            
            pre_idx, post_idx = int(res['pre'] * step), int(res['post'] * step)
            win = int(len(df) * 0.1)

            # 4. 수치 계산 (SyntaxError 발생 지점 수정 완료)
            v_pre = df.iloc[pre_idx : pre_idx + win][targets].mean().mean()
            v_post = df.iloc[post_idx : post_idx + win][targets].mean().mean()
            rate = ((v_post - v_pre) / v_pre) * 100 if v_pre != 0 else 0
