import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
import json

# [중요] 키를 코드에 직접 적지 않고, Streamlit의 보안 설정(Secrets)에서 불러옵니다.
if "GEMINI_API_KEY" in st.secrets:
    API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
else:
    st.error("⚠️ Streamlit 설정에서 API 키를 입력해주세요.")
    st.stop()

st.set_page_config(page_title="TrueWindow Neuro-AI", layout="wide")
st.title("🧠 TrueWindow Neuro-AI Analyzer")

uploaded_file = st.file_uploader("CSV 파일을 업로드하세요", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file, encoding='utf-8', skiprows=1)
        targets = ['Alpha_TP9', 'Alpha_AF7', 'Alpha_AF8', 'Alpha_TP10']
        
        with st.spinner("AI 분석 중..."):
            trend = df[targets].mean(axis=1).fillna(0).tolist()
            step = max(1, len(trend) // 100)
            summary = trend[::step]
            
            prompt = f"EEG Data: {summary}. Find baseline(0-30) and peak(40-100). Return ONLY JSON: {{\"pre\": 10, \"post\": 85}}"
            response = model.generate_content(prompt)
            res = json.loads(response.text.replace('```json', '').replace('```', '').strip())
            
            pre_idx, post_idx = int(res['pre'] * step), int(res['post'] * step)
            win = int(len(df) * 0.1)
            
            v_pre = df.iloc[pre_idx : pre_idx + win][targets].mean().mean()
            v_post = df.iloc[post_idx : post_idx + win][targets].mean().mean()
            rate = ((v_post - v_pre) / v_pre) * 100 if v_pre != 0 else 0

            st.success(f"✅ 분석 완료: Alpha파 변화율 {rate:+.2f}%")
            st.metric("최종 변화율", f"{rate:+.2f}%")
            st.line_chart(df[targets].mean(axis=1).rolling(window=100).mean())
    except Exception as e:
        st.error(f"오류 발생: {e}")
