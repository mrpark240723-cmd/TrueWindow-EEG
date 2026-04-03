import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
import json

# 1. AI 엔진 자동 탐색 (모든 모델 경로 시도)
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    
    @st.cache_resource
    def get_working_model():
        """내 API 키로 사용 가능한 모델을 리스트업하고, 작동하는 첫 번째 모델 반환"""
        try:
            # 서버에서 지원하는 모델 목록을 직접 가져옴
            available_models = [m.name for m in genai.list_models() 
                                if 'generateContent' in m.supported_generation_methods]
            
            # 1. 우선순위 모델들 시도 (Flash -> Pro 순서)
            for target in ['gemini-1.5-flash', 'gemini-pro', 'gemini-1.0-pro']:
                for am in available_models:
                    if target in am:
                        return genai.GenerativeModel(am)
            
            # 2. 정 안되면 목록 중 가장 첫 번째 모델이라도 선택
            if available_models:
                return genai.GenerativeModel(available_models[0])
        except Exception as e:
            st.error(f"모델 리스트 확보 실패: {e}")
        return None

    model = get_working_model()
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

        # 3. AI 분석 및 수치 계산
        if model:
            with st.spinner(f"AI({model.model_name})가 분석 구간을 탐색 중..."):
                trend = df[targets].mean(axis=1).fillna(0).tolist()
                step = max(1, len(trend) // 100)
                summary = trend[::step]
                
                prompt = f"EEG: {summary}. Find baseline(0-30) and peak(40-100). Return ONLY JSON: {{\"pre\": 10, \"post\": 80}}"
                response = model.generate_content(prompt)
                
                # 결과 파싱 (JSON 클렌징)
                txt = response.text.replace('```json', '').replace('```', '').strip()
                res = json.loads(txt)
                
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
        else:
            st.error("사용 가능한 모델을 찾지 못했습니다. API 키 권한을 확인해주세요.")

    except Exception as e:
        st.error(f"오류 발생: {e}")
