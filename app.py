import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
import json

# 1. AI 엔진 설정 (모델 명칭 및 설정 최적화)
API_KEY = "AIzaSyBVDBbXn_LTyrch5YmXDN1XER0-Uvc67KU"
genai.configure(api_key=API_KEY)

# 가장 호환성이 높은 gemini-1.5-flash-latest 또는 gemini-pro 사용
model = genai.GenerativeModel('gemini-1.5-flash-latest')

st.set_page_config(page_title="TrueWindow Neuro-AI Engine", layout="wide")

st.title("🧠 TrueWindow Neuro-AI Analysis Engine")
st.subheader("Gemini AI 기반 정밀 구간 탐색 시스템")

uploaded_file = st.file_uploader("분석할 MUSE 2 CSV 파일을 업로드하세요", type=["csv"])

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
            'Beta': ['Beta_TP9', 'Beta_AF7', 'Beta_AF8', 'Beta_TP10'],
            'Theta': ['Theta_TP9', 'Theta_AF7', 'Theta_AF8', 'Theta_TP10'],
            'Delta': ['Delta_TP9', 'Delta_AF7', 'Delta_AF8', 'Delta_TP10']
        }

        for cat in targets.values():
            for col in cat:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna(subset=['Alpha_TP9'], how='all').reset_index(drop=True)

        # 3. AI 분석용 데이터 요약 (Gemini 전달용)
        alpha_trend = df[targets['Alpha']].mean(axis=1).fillna(0).tolist()
        step = max(1, len(alpha_trend) // 200)
        summarized_trend = alpha_trend[::step]

        # 4. Gemini AI 구간 탐색 (오류 방지 로직 포함)
        with st.spinner("Gemini AI가 최적 분석 구간을 산출하고 있습니다..."):
            prompt = f"""
            뇌파 데이터 트렌드 분석: {summarized_trend}
            
            위 데이터를 분석하여 다음 구간의 index(0~200 사이)를 JSON으로 응답하세요.
            1. pre_start/pre_end: 데이터 초반부의 가장 안정적인 낮은 구간
            2. post_start/post_end: 데이터 후반부의 가장 활성화된 높은 구간
            응답 형식: {{"pre_start": 0, "pre_end": 40, "post_start": 120, "post_end": 180}}
            """
            
            # 모델 호출 (안전한 생성 설정 사용)
            response = model.generate_content(
                prompt,
                generation_config={"response_mime_type": "application/json"}
            )
            
            indices = json.loads(response.text)

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
            valid_cols = [c for c in cols if c in df.columns]
            v_pre = pre_df[valid_cols].mean().mean()
            v_post = post_df[valid_cols].mean().mean()
            rate = ((v_post - v_pre) / v_pre) * 100 if v_pre != 0 else 0
            
            with m_cols[i]:
                st.metric(label=f"{name} 변동률", value=f"{rate:+.2f}%", delta=f"{v_post-v_pre:.4f}")
            
            report_list.append({"지표": name, "시청전(Baseline)": round(v_pre, 5), "시청후(Peak)": round(v_post, 5), "증감률": f"{rate:+.2f}%"})

        st.divider()
        st.write("### 📑 AI 정밀 데이터 분석표")
        st.table(pd.DataFrame(report_list))
        st.line_chart(df[targets['Alpha']].mean(axis=1).rolling(window=100).mean())

    except Exception as e:
        st.error(f"분석 엔진 구동 오류: {e}")
