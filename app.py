import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
import json

# 1. AI 엔진 자동 탐색 (모든 모델 경로 시도)
API_KEY = "AIzaSyAtZ66sTetZ1pweRevHov9Z53rf5A_fq0s"
genai.configure(api_key=API_KEY)

@st.cache_resource
def get_working_model():
    """내 API 키로 사용 가능한 모델을 리스트업하고, 작동하는 첫 번째 모델 반환"""
    try:
        available_models = [m.name for m in genai.list_models() 
                            if 'generateContent' in m.supported_generation_methods]
        
        # 1. 우선순위 모델들 시도 (Flash -> Pro)
        for target in ['gemini-1.5-flash', 'gemini-pro', 'gemini-1.0-pro']:
            for am in available_models:
                if target in am:
                    return genai.GenerativeModel(am)
        
        # 2. 아무 모델이나 잡히는 대로 시도
        if available_models:
            return genai.GenerativeModel(available_models[0])
    except Exception as e:
        st.error(f"모델 리스트 확보 실패: {e}")
    return None

model = get_working_model()

st.set_page_config(page_title="TrueWindow Neuro-AI", layout="wide")
st.title("🧠 TrueWindow Neuro-AI Analyzer")

uploaded_file = st.file_uploader("MUSE 2 CSV 파일을 업로드하세요", type=["csv"])

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
            'Beta': ['Beta_TP9', 'Beta_AF7', 'Beta_AF8', 'Beta_TP10']
        }

        for cat in targets.values():
            for col in cat:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna(subset=['Alpha_TP9'], how='all').reset_index(drop=True)

        # 3. 데이터 요약
        alpha_trend = df[targets['Alpha']].mean(axis=1).fillna(0).tolist()
        step = max(1, len(alpha_trend) // 120)
        summarized = alpha_trend[::step]

        # 4. AI 분석
        if model:
            with st.spinner(f"AI({model.model_name})가 분석 구간을 탐색 중..."):
                prompt = f"Analyze EEG. Data: {summarized}. Return ONLY JSON: {{\"pre_s\": 0, \"pre_e\": 20, \"post_s\": 70, \"post_e\": 95}}"
                response = model.generate_content(prompt)
                txt = response.text.replace('```json', '').replace('```', '').strip()
                idx = json.loads(txt)

                pre_s, pre_e = int(idx['pre_s'] * step), int(idx['pre_e'] * step)
                post_s, post_e = int(idx['post_s'] * step), int(idx['post_e'] * step)

            # 5. 결과 계산
            pre_df, post_df = df.iloc[pre_s:pre_e], df.iloc[post_s:post_e]
            st.success(f"✅ 분석 완료: Baseline({pre_s}~{pre_e}) vs Peak({post_s}~{post_e})")

            m_cols = st.columns(2)
            for i, (name, cols) in enumerate(targets.items()):
                exist_cols = [c for c in cols if c in df.columns]
                v_pre = pre_df[exist_cols].mean().mean()
                v_post = post_df[exist_cols].mean().mean()
                rate = ((v_post - v_pre) / v_pre) * 100 if v_pre != 0 else 0
                with m_cols[i]:
                    st.metric(label=f"{name} 변동률", value=f"{rate:+.2f}%", delta=f"{v_post-v_pre:.4f}")

            st.divider()
            st.line_chart(df[targets['Alpha']].mean(axis=1).rolling(window=100).mean())
        else:
            st.error("사용 가능한 AI 모델이 없습니다. API 키를 확인해주세요.")

    except Exception as e:
        st.error(f"오류 발생: {e}")
