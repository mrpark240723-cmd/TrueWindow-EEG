import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
import json

# 1. AI 엔진 설정 (Gemini API)
API_KEY = "AIzaSyBVDBbXn_LTyrch5YmXDN1XER0-Uvc67KU"
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

st.set_page_config(page_title="TrueWindow Neuro-AI Engine (Powered by Gemini)", layout="wide")

st.title("🧠 TrueWindow Neuro-AI Analysis Engine")
st.subheader("Gemini AI 기반 자동 구간 탐색 및 정밀 분석 시스템")

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

        # 3. AI에게 보낼 데이터 요약 (Gemini가 맥락을 읽을 수 있도록 200개 포인트로 압축)
        alpha_trend = df[targets['Alpha']].mean(axis=1).fillna(0).tolist()
        step = max(1, len(alpha_trend) // 200)
        summarized_trend = alpha_trend[::step]

        # 4. Gemini AI에게 최적 구간 탐색 요청
        with st.spinner("Gemini AI가 데이터의 패턴을 분석하여 최적의 비교 구간을 찾고 있습니다..."):
            prompt = f"""
            당신은 뇌파 분석 전문가입니다. 아래는 피험자의 시청 전/후가 합쳐진 Alpha파 데이터 트렌드(200포인트 요약)입니다.
            데이터: {summarized_trend}
            
            1. 데이터의 앞부분에서 가장 안정적이고 낮은 Baseline(시청 전) 구간의 인덱스 범위를 찾으세요.
            2. 데이터의 뒷부분에서 시청 효과가 극대화된 가장 높은 Peak(시청 중) 구간의 인덱스 범위를 찾으세요.
            3. 결과를 오직 JSON 형식으로만 응답하세요. 예: {{"pre_start": 0, "pre_end": 40, "post_start": 120, "post_end": 180}}
            """
            response = model.generate_content(prompt)
            indices = json.loads(response.text.strip().replace('```json', '').replace('```', ''))

            # 실제 데이터 인덱스로 환산
            pre_s, pre_e = indices['pre_start'] * step, indices['pre_end'] * step
            post_s, post_e = indices['post_start'] * step, indices['post_end'] * step

        # 5. 정밀 구간 계산 (AI가 지정한 구간)
        pre_df = df.iloc[pre_s : pre_e]
        post_df = df.iloc[post_s : post_e]

        st.success(f"✅ Gemini AI 분석 완료: Baseline({pre_s}~{pre_e}) 및 Peak({post_s}~{post_e}) 구간 탐지")

        # 6. 결과 출력
        m_cols = st.columns(4)
        report_list = []

        for i, (name, cols) in enumerate(targets.items()):
            valid_cols = [c for c in cols if c in df.columns]
            v_pre = pre_df[valid_cols].mean().mean()
            v_post = post_df[valid_cols].mean().mean()
            rate = ((v_post - v_pre) / v_pre) * 100 if v_pre != 0 else 0
            
            with m_cols[i]:
                st.metric(label=f"{name} 지표", value=f"{rate:+.2f}%", delta=f"{v_post-v_pre:.4f}")
            
            report_list.append({"지표": name, "시청전(AI탐색)": round(v_pre, 5), "시청후(AI탐색)": round(v_post, 5), "변화율": f"{rate:+.2f}%"})

        st.divider()
        
        # 7. 리포트 및 그래프
        c1, c2 = st.columns([1, 2])
        with c1:
            st.write("### 📑 AI 정밀 리포트")
            st.table(pd.DataFrame(report_list))
            st.caption("※ 본 수치는 Gemini AI가 데이터의 노이즈를 식별하고 최적 구간을 직접 선정한 결과입니다.")
        with c2:
            st.write("### 📈 뇌파 흐름 및 AI 분석 영역")
            st.line_chart(df[targets['Alpha']].mean(axis=1).rolling(window=100).mean())

    except Exception as e:
        st.error(f"AI 분석 엔진 구동 중 오류가 발생했습니다: {e}")
