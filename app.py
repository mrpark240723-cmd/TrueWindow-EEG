import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="TrueWindow Neuro-AI Analyzer", layout="wide")

st.title("🪟 TrueWindow Real-Time Neuro-Analyzer")
st.info("💡 데이터 내 노이즈를 제거하고 실시간 전수 조사를 실시합니다.")

uploaded_file = st.file_uploader("분석할 CSV 파일을 업로드하세요", type=["csv"])

if uploaded_file:
    try:
        # 1. 인코딩 처리 및 데이터 로드 (첫 줄 타이틀 건너뛰기)
        try:
            df = pd.read_csv(uploaded_file, encoding='cp949', skiprows=1)
        except:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding='utf-8', skiprows=1)
        
        # 2. 분석 지표 설정
        targets = {
            'Alpha (이완)': ['Alpha_TP9', 'Alpha_AF7', 'Alpha_AF8', 'Alpha_TP10'],
            'Beta (집중)': ['Beta_TP9', 'Beta_AF7', 'Beta_AF8', 'Beta_TP10'],
            'Theta (명상)': ['Theta_TP9', 'Theta_AF7', 'Theta_AF8', 'Theta_TP10'],
            'Delta (회복)': ['Delta_TP9', 'Delta_AF7', 'Delta_AF8', 'Delta_TP10']
        }

        # [핵심] 문자열 오류 방지: 모든 데이터를 숫자로 변환 (변환 안 되는 문자는 NaN 처리)
        for cat in targets.values():
            for col in cat:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

        # 결측치(NaN)가 있는 행은 계산에서 제외
        df = df.dropna(subset=[c for sublist in targets.values() for c in sublist if c in df.columns], how='all')

        # 3. 데이터 분할 (전/후)
        mid = len(df) // 2
        pre_df = df.iloc[:mid]
        post_df = df.iloc[mid:]

        st.success(f"✅ 노이즈 제거 후 {len(df):,}행의 유효 데이터를 분석했습니다.")

        # 4. 결과 출력
        cols = st.columns(4)
        results_list = []

        for i, (name, col_list) in enumerate(targets.items()):
            valid_cols = [c for c in col_list if c in df.columns]
            
            if valid_cols:
                # 숫자 데이터만 평균 계산
                pre_val = pre_df[valid_cols].mean().mean()
                post_val = post_df[valid_cols].mean().mean()
                
                diff = post_val - pre_val
                rate = (diff / pre_val) * 100 if pre_val != 0 else 0
                
                with cols[i]:
                    st.metric(label=name, value=f"{rate:+.1f}%", delta=f"{diff:.3f}")
                
                results_list.append({
                    "지표": name.split(' ')[0],
                    "시청 전 평균": f"{pre_val:.4f}",
                    "시청 후 평균": f"{post_avg if 'post_avg' in locals() else post_val:.4f}",
                    "증감률": f"{rate:+.2f}%"
                })

        st.divider()

        # 5. 시각화
        c1, c2 = st.columns([1, 1.5])
        with c1:
            st.write("### 📑 실시간 데이터 통계")
            st.table(pd.DataFrame(results_list))
        with c2:
            st.write("### 📈 뇌파 변화 추이")
            trend_cols = [c for c in ['Alpha_TP9', 'Beta_TP9', 'Theta_TP9', 'Delta_TP9'] if c in df.columns]
            if trend_cols:
                st.line_chart(df[trend_cols].tail(3000))

    except Exception as e:
        st.error(f"데이터 분석 오류: {e}")
