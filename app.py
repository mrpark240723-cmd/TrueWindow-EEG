import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
import json

# 1. Streamlit Secrets에서 API 키 불러오기
if "GEMINI_API_KEY" in st.secrets:
    API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
else:
    st.error("⚠️ Streamlit 설정(Secrets)에서 API 키를 입력해주세요.")
    st.stop()

st.set_page_config(page_title="TrueWindow Neuro-AI", layout="wide")
st.title("🧠 TrueWindow Neuro-AI Analyzer")

uploaded_file = st.file_uploader("MUSE 2 CSV 파일을 업로드하세요", type=["csv"])

if uploaded_file:
    try:
        # 2. 인코딩 오류 방지용 데이터 로드 로직
        df = None
        # 시도 1: utf-8
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding='utf-8', skiprows=1)
        except UnicodeDecodeError:
            # 시도 2: cp949 (한국어 윈도우 표준 - 0xbd 오류 해결)
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding='cp949', skiprows=1)

        # 3. 데이터 분석 (Alpha파 기준)
        targets = ['Alpha_TP9', 'Alpha_AF7', 'Alpha_AF8', 'Alpha_TP10']
        
        # 숫자형 변환 및 정제
        for col in targets:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 유효 데이터 필터링
        df = df.dropna(subset=['Alpha_TP9']).reset_index(drop=True)

        if not df.empty:
            with st.spinner("AI가 최적 분석 구간을 산출 중입니다..."):
                # 요약 데이터 생성 (100포인트)
                trend = df[targets].mean(axis=1).fillna(0).tolist()
                step = max(1, len(trend) // 100)
                summary = trend[::step]
                
                # AI에게 구간 탐색 요청
                prompt = f"Analyze EEG: {summary}. Find baseline(0-30) and peak(40-100). Return ONLY JSON: {{\"pre\": 10, \"post\": 80}}"
                response = model.generate_content(prompt)
                
                # 결과 파싱
                res_json = json.loads(response.text.replace('```json', '').replace('```', '').strip())
                pre_idx, post_idx = int(res_json['pre'] * step), int(res_json['post'] * step)
                win = int(len(df) * 0.1) # 10% 구간을 윈도우로 설정

                # 수치 계산
                v_pre = df.iloc[pre_idx : pre_idx + win][targets].mean().mean()
                v_post = df.iloc[post_idx :
