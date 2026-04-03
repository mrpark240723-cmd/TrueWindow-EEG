import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
import json

# 1. API 설정
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    @st.cache_resource
    def get_model():
        try:
            models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
            for t in ['gemini-1.5-flash', 'gemini-pro']:
                for m in models:
                    if t in m: return genai.GenerativeModel(m)
            return genai.GenerativeModel(models[0]) if models else None
        except: return None
    model = get_model()
else:
    st.error("⚠️ Streamlit Secrets 설정을 확인해주세요.")
    st.stop()

st.title("🧠 TrueWindow Neuro-AI Analyzer")

uploaded_file = st.file_uploader("MUSE 2 CSV 파일을 업로드하세요", type=["csv"])

if uploaded_file and model:
    try:
        # 2. 데이터 로드
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

        # 3. AI 정밀 분석 (들여쓰기 수정 완료)
        with st.spinner(f"AI({model.model_name})가 최저-최고점을 탐색 중..."):
            # 분석용 데이터 요약
            trend = df[targets].mean(axis=1).fillna(0).tolist()
            step = max(1, len(trend) // 120)
            summary = trend[::step]
            
            # AI에게 '가장 낮은 안정기'를 찾으라고 강력하게 지시
            prompt = f"""
            EEG Data: {summary}
            Task: Find the MOST STABLE MINIMUM point (Baseline) in the first 40% and the HIGHEST PEAK (Concentration) in the last 50%.
            Return ONLY JSON: {{"pre": baseline_index, "post": peak_index}}
            """
            
            response = model.generate_content(prompt)
            res = json.loads(response.text.replace('```json', '').replace('```', '').strip())
            
            p_idx, q_idx = int(res['pre'] * step), int(res['post'] * step)
            win = int(len(df) * 0.05) # 5% 정밀 구간

            v_pre = df.iloc[p_idx : p_idx + win][targets].mean().mean()
            v_post = df.iloc[q_idx : q_idx + win][targets].mean().mean()
            rate = ((v_post - v_pre) / v_pre) * 100 if v_pre != 0 else 0

            # 4. 결과 출력
            st.success(f"✅ 분석 완료 (모델: {model.model_name})")
            st.metric("Alpha파 변화율 (최저점 대비)", f"{rate:+.2f}%", f"{v_post-v_pre:.4f}")
            
            st.write("### 📈 뇌파 흐름 및 분석 구간")
            st.line_chart(df[targets].mean(axis=1).rolling(window=100).mean())
            
            st.table(pd.DataFrame([{
                "구분": "Alpha파",
                "안정기(최저점)": round(v_pre, 4),
                "몰입기(최고점)": round(v_post, 4),
                "변화율": f"{rate:+.2f}%"
            }]))

    except Exception as e:
        st.error(f"분석 중 오류 발생: {e}")
