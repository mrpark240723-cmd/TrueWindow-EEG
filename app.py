import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
import json

# 1. AI 엔진 설정 (문법 오류 수정 및 안정화)
API_KEY = "AIzaSyBVDBbXn_LTyrch5YmXDN1XER0-Uvc67KU"
genai.configure(api_key=API_KEY)

def load_ai_model():
    # 시도 1: 가장 표준적인 명칭
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        return model
    except:
        # 시도 2: 경로 포함 명칭 (환경에 따른 대응)
        try:
            return genai.GenerativeModel('models/gemini-1.5-flash')
        except Exception as e:
            return None

model = load_ai_model()

st.set_page_config(page_title="TrueWindow Neuro-AI Analyzer", layout="wide")

st.title("🧠 TrueWindow Neuro-AI Analysis Engine")
st.subheader("Gemini AI 기반 정밀 구간 탐색 시스템")

uploaded_file = st.file_uploader("분석할 MUSE 2 CSV 파일을 업로드하세요", type=["csv"])

if uploaded_file:
    try:
        # 2. 데이터 로드
        try:
            df = pd.read_csv(uploaded_file, encoding='cp949', skiprows=1)
        except:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding='utf-8', skiprows=1)

        targets = {
            'Alpha': ['Alpha_TP9', 'Alpha_AF7', 'Alpha_AF8', 'Alpha_TP10'],
            'Beta': ['Beta_TP9', 'Beta_AF7', 'Beta_AF8', 'Beta_TP10'],
            'Theta': ['Theta_TP9', 'Theta_AF7', 'Theta_AF8', 'Theta_TP10'],
            'Delta': ['Delta_TP9', 'Delta_AF7', 'Delta_AF8', 'Delta_TP10']
        }

        for cat in targets.values():
            for col in cat:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna(subset=['Alpha_TP9'], how='all').reset_index(drop=True)

        # 3. 데이터 요약
        alpha_trend = df[targets['Alpha']].mean(axis=1).fillna(0).tolist()
        step = max(1, len(alpha_trend) // 150)
        summarized_trend = alpha_trend[::step]

        # 4. AI 분석
        if model:
            with st.spinner("Gemini AI가 데이터를 분석 중입니다..."):
                prompt = f"Analyze this EEG: {summarized_trend}. Return ONLY JSON: {{\"pre_start\": 0, \"pre_end\": 30, \"post_start\": 100, \"post_end\": 145}}"
                response = model.generate_content(prompt)
                
                # 텍스트 클렌징 및 파싱
                txt = response.text.replace('```json', '').replace('```', '').strip()
                indices = json.loads(txt)

                pre_s, pre_e = int(indices['pre_start'] * step), int(indices['pre_end'] * step)
                post_s, post_e = int(indices['post_start'] * step), int(indices['post_end'] * step)

            # 5. 수치 계산
            pre_df, post_df = df.iloc[pre_s : pre_e], df.iloc[post_s : post_e]
            
            st.success(f"✅ 분석 완료: Baseline({pre_s}~{pre_e}) vs Peak({post_s}~{post_e})")

            m_cols = st.columns(4)
            report_list = []

            for i, (name, cols) in enumerate(targets.items()):
                exist_cols = [c for c in cols if c in df.columns]
                v_pre = pre_df[exist_cols].mean().mean()
                v_post = post_df[exist_cols].mean().mean()
                rate = ((v_post - v_pre) / v_pre) * 100 if v_pre != 0 else 0
                
                with m_cols[i]:
                    is_pos = (name != 'Beta' and rate > 0) or (name == 'Beta' and rate < 0)
                    st.metric(
