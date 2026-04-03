import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
import json

# 1. AI 엔진 설정 (가장 표준적인 모델 경로 적용)
API_KEY = "AIzaSyBVDBbXn_LTyrch5YmXDN1XER0-Uvc67KU"
genai.configure(api_key=API_KEY)

# 'models/' 접두사를 붙여 경로를 명확히 지정합니다.
model = genai.GenerativeModel('models/gemini-1.5-flash')

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

        # 3. AI 분석용 데이터 요약 (150포인트로 최적화)
        alpha_trend = df[targets['Alpha']].mean(axis=1).fillna(0).tolist()
        step = max(1, len(alpha_trend) // 150)
        summarized_trend = alpha_trend[::step]

        # 4. Gemini AI 구간 탐색
        with st.spinner("Gemini AI가 최적 분석 구간을 산출하고 있습니다..."):
            # 시스템 프롬프트를 명확히 전달
            prompt = f"""
            You are an EEG analysis expert. Analyze the following Alpha wave trend data (150 points):
            Data: {summarized_trend}
            
            Find the optimal indices (0-150) for:
            1. pre_start/pre_end: The most stable baseline in the early part.
            2. post_start/post_end: The highest peak during the stimulus in the later part.
            
            Respond ONLY in JSON format like this:
            {{"pre_start": 0, "pre_end": 30, "post_start": 100, "post_end": 140}}
            """
            
            # 모델 생성 방식 보완
            response = model.generate_content(prompt)
            
            # JSON 텍스트 정제 후 로드
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
            valid_cols = [c in df.columns for c in cols]
            # 실제 존재하는 컬럼 필터링
            exist_cols = [c for c in cols if c in df.columns]
            
            v_pre = pre_df[exist_cols].mean().mean()
            v_post = post_df[exist_cols].mean().mean()
            rate = ((v_post - v_pre) / v_pre) * 100 if v_pre != 0 else 0
            
            with m_cols[i]:
                # Beta파는 하락이 긍정적이므로 색상 반전
                is_pos = (name != 'Beta' and rate > 0) or (name == 'Beta' and rate < 0)
                st.metric(label=f"{name} 변동률", value=f"{rate:+.2f}%", 
                          delta=f"{v_post-v_pre:.4f}", delta_color="normal" if is_pos else "inverse")
            
            report_list.append({"지표": name, "시청전(Baseline)": round(v_pre, 5), "시청후(Peak)": round(v_post, 5), "증감률": f"{rate:+.2f}%"})

        st.divider()
        st.write("### 📑 AI 정밀 데이터 분석표")
        st.table(pd.DataFrame(report_list))
        st.line_chart(df[targets['Alpha']].mean(axis=1).rolling(window=100).mean())

    except Exception as e:
        st.error(f"분석 엔진 구동 오류: {e}")
