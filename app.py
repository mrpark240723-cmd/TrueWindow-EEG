import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
import json

# 1. AI 엔진 설정 (이중 경로 안전장치 적용)
API_KEY = "AIzaSyBVDBbXn_LTyrch5YmXDN1XER0-Uvc67KU"
genai.configure(api_key=API_KEY)

def get_gemini_model():
    # 시도 1: 표준 명칭
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        # 모델이 작동하는지 테스트 호출 (가벼운 인사)
        model.generate_content("test")
        return model
    except:
        # 시도 2: 경로 포함 명칭 (404 방지용)
        try:
            return genai.GenerativeModel('models/gemini-1.5-flash')
        except Exception as e:
            st.error(f"모델 로드 실패: {e}")
            return None

model = get_gemini_model()

st.set_page_config(page_title="TrueWindow Neuro-AI Engine", layout="wide")

st.title("🧠 TrueWindow Neuro-AI Analysis Engine")
st.subheader("Gemini AI 기반 정밀 구간 탐색 시스템")

uploaded_file = st.file_uploader("분석할 MUSE 2 CSV 파일을 업로드하세요", type=["csv"])

if uploaded_file:
    try:
        # 2. 데이터 로드 및 전처리
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

        # 3. AI 분석용 데이터 요약 (150포인트)
        alpha_trend = df[targets['Alpha']].mean(axis=1).fillna(0).tolist()
        step = max(1, len(alpha_trend) // 150)
        summarized_trend = alpha_trend[::step]

        # 4. Gemini AI 구간 탐색
        if model:
            with st.spinner("Gemini AI가 최적 분석 구간을 산출 중입니다..."):
                prompt = f"""
                Analyze this EEG Alpha trend (150 points): {summarized_trend}
                Find indices (0-150) for:
                1. pre_start/pre_end: stable baseline (early part)
                2. post_start/post_end: highest peak (later part)
                Respond ONLY in JSON: {{"pre_start": 0, "pre_end": 30, "post_start": 100, "post_end": 140}}
                """
                response = model.generate_content(prompt)
                
                # 마크다운 제거 및 JSON 파싱
                clean_json = response.text.replace('```json', '').replace('```', '').strip()
                indices = json.loads(clean_json)

                pre_s, pre_e = int(indices['pre_start'] * step), int(indices['pre_end'] * step)
                post_s, post_e = int(indices['post_start'] * step), int(indices['post_end'] * step)

            # 5. 최종 데이터 계산
            pre_df = df.iloc[pre_s : pre_e]
            post_df = df.iloc[post_s : post_e]

            st.success(f"✅ AI 구간 분석 완료: Baseline({pre_s}~{pre_e}) vs Peak({post_s}~{post_e})")

            # 6. 결과 시각화
            m_cols = st.columns(4)
            report_list = []

            for i, (name, cols) in enumerate(targets.items()):
                exist_cols = [c for c in cols if c in df.columns]
                v_pre = pre_df[exist_cols].mean().mean()
                v_post = post_df[exist_cols].mean().mean()
                rate = ((v_post - v_pre) / v_pre) * 100 if v_pre != 0 else 0
                
                with m_cols[i]:
                    is_pos = (name != 'Beta' and rate > 0) or (name == 'Beta' and rate < 0)
                    st.metric(label=f"{name} 변동률", value=f"{rate:+.2f}%", 
                              delta=f"{v_post-v_pre:.4f}", delta_color="normal" if is_pos else "inverse")
                
                report_list.append({"지표": name, "시청전(Baseline)": round(v_pre, 5), "시청후(Peak)": round(v_post, 5), "증감률": f"{rate:+.2f}%"})

            st.divider()
            st.write("### 📑 AI 정밀 데이터 분석표")
            st.table(pd.DataFrame(report_list))
            st.line_chart(df[targets['Alpha']].mean(axis=1).rolling(window=100).mean())
        else:
            st.error("AI 모델을 로드할 수 없습니다. API 키 상태를 확인해 주세요.")

    except Exception as e:
        st.error(f"분석 엔진 오류: {e}")
