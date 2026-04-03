import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
import json

# 1. API 설정
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    
    @st.cache_resource
    def get_working_model():
        try:
            # 시도 1: 서버에서 모델 리스트 직접 조회
            models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
            for target in ['gemini-1.5-flash', 'gemini-pro']:
                for m in models:
                    if target in m: return genai.GenerativeModel(m)
            
            # 시도 2: 리스트 조회 실패 시 가장 표준적인 이름으로 강제 생성
            return genai.GenerativeModel('gemini-1.5-flash')
        except Exception as e:
            # 최종 방어선: 에러 발생 시 기본 모델 반환
            return genai.GenerativeModel('gemini-1.5-flash')

    model = get_working_model()
else:
    st.error("⚠️ Streamlit Settings > Secrets에 GEMINI_API_KEY를 설정해주세요.")
    st.stop()

st.title("🧠 TrueWindow Neuro-AI Analyzer")

uploaded_file = st.file_uploader("MUSE 2 CSV 파일을 업로드하세요", type=["csv"])

if uploaded_file and model:
    try:
        # 데이터 로드 (인코딩 대응)
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

        if not df.empty:
            with st.spinner(f"AI({model.model_name})가 분석 중입니다..."):
                trend = df[targets].mean(axis=1).fillna(0).tolist()
                step = max(1, len(trend) // 100)
                
                prompt = f"EEG: {trend[::step]}. Find baseline(0-30) and peak(40-100). Return ONLY JSON: {{\"pre\": 10, \"post\": 80}}"
                response = model.generate_content(prompt)
                res = json.loads(response.text.replace('```json', '').replace('```', '').strip())
                
                p_idx, q_idx = int(res['pre'] * step), int(res['post'] * step)
                win = int(len(df) * 0.1)

                v_pre = df.iloc[p_idx : p_idx + win][targets].mean().mean()
                v_post = df.iloc[q_idx : q_idx + win][targets].mean().mean()
                rate = ((v_post - v_pre) / v_pre) * 100 if v_pre != 0 else 0

                st.success(f"✅ 분석 완료 (모델: {model.model_name})")
                st.metric("Alpha파 변화율", f"{rate:+.2f}%", f"{v_post-v_pre:.4f}")
                st.line_chart(df[targets].mean(axis=1).rolling(window=100).mean())
        else:
            st.warning("분석할 유효한 데이터가 없습니다.")
            
    except Exception as e:
        # 여기서 403이나 404가 나오면 API 키의 권한 문제입니다.
        st.error(f"AI 분석 중 오류 발생: {e}")
