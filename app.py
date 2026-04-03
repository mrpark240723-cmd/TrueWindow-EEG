import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
import json

# 1. API 설정 및 모델 자동 탐색
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    
    # 404 오류 방지: 사용 가능한 모델명을 자동으로 찾음
    @st.cache_resource
    def get_model():
        try:
            # 서버에서 지원하는 모델 목록을 가져와 gemini-1.5-flash가 포함된 이름을 선택
            for m in genai.list_models():
                if 'gemini-1.5-flash' in m.name and 'generateContent' in m.supported_generation_methods:
                    return genai.GenerativeModel(m.name)
            # 못 찾을 경우 기본형 시도
            return genai.GenerativeModel('gemini-1.5-flash')
        except:
            return genai.GenerativeModel('gemini-pro')

    model = get_model()
else:
    st.error("⚠️ Streamlit Settings > Secrets에 GEMINI_API_KEY를 설정해주세요.")
    st.stop()

st.set_page_config(page_title="TrueWindow Neuro-AI", layout="wide")
st.title("🧠 TrueWindow Neuro-AI Analyzer")

uploaded_file = st.file_uploader("MUSE 2 CSV 파일을 업로드하세요", type=["csv"])

if uploaded_file:
    try:
        # 2. 데이터 로드 (인코딩 대응)
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
            
            prompt = f"EEG: {summary}. Find baseline(0-30) and peak(40-100). Return ONLY JSON: {{\"pre\": 10, \"post\": 85}}"
            response = model.generate_content(prompt)
            res = json.loads(response.text.replace('```json', '').replace('```', '').strip())
            
            p_idx, q_idx = int(res['pre'] * step), int(res['post'] * step)
            win = int(len(df) * 0.1)

            v_pre = df.iloc[p_idx : p_idx + win][targets].mean().mean()
            v_post = df.iloc[q_idx : q_idx + win][targets].mean().mean()
            rate = ((v_post - v_pre) / v_pre) * 100 if v_pre != 0 else 0

            # 4. 결과 출력
            st.success(f"✅ 분석 완료 (사용 모델: {model.model_name})")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Alpha파 변화율", f"{rate:+.2f}%", f"{v_post-v_pre:.4f}")
            with col2:
                st.line_chart(df[targets].mean(axis=1).rolling(window=100).mean())
            
            st.table(pd.DataFrame([{"지표": "Alpha", "Baseline": round(v_pre, 5), "Peak": round(v_post, 5), "변화율": f"{rate:+.2f}%"}]))

    except Exception as e:
        st.error(f"오류 발생: {e}")
