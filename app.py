import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
import json

# 1. AI 설정 (Gemini API)
API_KEY = "AIzaSyBVDBbXn_LTyrch5YmXDN1XER0-Uvc67KU"
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

st.set_page_config(page_title="TrueWindow Neuro-AI", layout="wide")
st.title("🧠 TrueWindow Neuro-AI Analyzer")

uploaded_file = st.file_uploader("MUSE 2 CSV 파일을 업로드하세요", type=["csv"])

if uploaded_file:
    try:
        # 2. 데이터 로드 및 정제
        try:
            df = pd.read_csv(uploaded_file, encoding='cp949', skiprows=1)
        except:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding='utf-8', skiprows=1)

        targets = {
            'Alpha': ['Alpha_TP9', 'Alpha_AF7', 'Alpha_AF8', 'Alpha_TP10'],
            'Beta': ['Beta_TP9', 'Beta_AF7', 'Beta_AF8', 'Beta_TP10']
        }

        for cat in targets.values():
            for col in cat:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna(subset=['Alpha_TP9'], how='all').reset_index(drop=True)

        # 3. AI 분석용 요약 (100포인트)
        alpha_trend = df[targets['Alpha']].mean(axis=1).fillna(0).tolist()
        step = max(1, len(alpha_trend) // 100)
        summarized = alpha_trend[::step]

        # 4. Gemini AI 구간 탐색
        with st.spinner("AI가 최적 구간을 분석 중입니다..."):
            prompt = f"Analyze EEG: {summarized}. Return ONLY JSON: {{\"pre_s\": 0, \"pre_e\": 20, \"post_s\": 70, \"post_e\": 95}}"
            response = model.generate_content(prompt)
            idx = json.loads(response.text.replace('```json', '').replace('```', '').strip())

            pre_s, pre_e = int(idx['pre_s'] * step), int(idx['pre_e'] * step)
            post_s, post_e = int(idx['post_s'] * step), int(idx['post_e'] * step)

        # 5. 결과 계산 및 출력
        pre_df, post_df = df.iloc[pre_s:pre_e], df.iloc[post_s:post_e]
        st.success(f"✅ 분석 구간: Baseline({pre_s}~{pre_e}) vs Peak({post_s}~{post_e})")

        m_cols = st.columns(2)
        for i, (name, cols) in enumerate(targets.items()):
            exist_cols = [c for c in cols if c in df.columns]
            v_pre = pre_df[exist_cols].mean().mean()
            v_post = post_df[exist_cols].mean().mean()
            rate = ((v_post - v_pre) / v_pre) * 100 if v_pre != 0 else 0
            
            with m_cols[i]:
                st.metric(label=f"{name} 변동", value=f"{rate:+.2f}%", delta=f"{v_post-v_pre:.4f}")

        st.divider()
        st.line_chart(df[targets['Alpha']].mean(axis=1).rolling(window=100).mean())

    except Exception as e:
        st.error(f"오류 발생: {e}")
