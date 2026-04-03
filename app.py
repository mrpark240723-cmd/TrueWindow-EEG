import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="TrueWindow Neuro-AI Analyzer", layout="wide")

st.title("🪟 TrueWindow Real-Time Neuro-Analyzer")
st.info("💡 업로드된 CSV의 수만 개 데이터를 실시간 전수 조사하여 분석 결과를 도출합니다.")

uploaded_file = st.file_uploader("분석할 CSV 파일을 업로드하세요", type=["csv"])

if uploaded_file:
    try:
        # 1. 인코딩 오류 해결: 한국어 엑셀(CP949)과 일반(UTF-8)을 모두 시도합니다.
        try:
            # 첫 줄(타이틀)을 건너뛰고 데이터 로드
            df = pd.read_csv(uploaded_file, encoding='cp949', skiprows=1)
        except:
            uploaded_file.seek(0) # 파일 포인터 초기화
            df = pd.read_csv(uploaded_file, encoding='utf-8', skiprows=1)
        
        # 2. 분석 지표 설정 (MUSE 2 데이터 컬럼 기준)
        targets = {
            'Alpha (이완)': ['Alpha_TP9', 'Alpha_AF7', 'Alpha_AF8', 'Alpha_TP10'],
            'Beta (집중/긴장)': ['Beta_TP9', 'Beta_AF7', 'Beta_AF8', 'Beta_TP10'],
            'Theta (명상)': ['Theta_TP9', 'Theta_AF7', 'Theta_AF8', 'Theta_TP10'],
            'Delta (회복)': ['Delta_TP9', 'Delta_AF7', 'Delta_AF8', 'Delta_TP10']
        }

        # 3. 실시간 데이터 분할 (합본 파일의 전/후 비교를 위해 절반으로 나눔)
        mid = len(df) // 2
        pre_df = df.iloc[:mid]
        post_df = df.iloc[mid:]

        st.success(f"✅ 총 {len(df):,}행의 데이터를 실시간 분석 중입니다.")

        # 4. 결과 출력
        cols = st.columns(4)
        results_list = []

        for i, (name, col_list) in enumerate(targets.items()):
            # 존재하는 컬럼만 선별
            valid_cols = [c for c in col_list if c in df.columns]
            
            if valid_cols:
                # 시청 전/후 평균값 계산 (전수 조사)
                pre_val = pre_df[valid_cols].mean().mean()
                post_val = post_df[valid_cols].mean().mean()
                
                # 변화율 계산
                diff = post_val - pre_val
                rate = (diff / pre_val) * 100 if pre_val != 0 else 0
                
                with cols[i]:
                    st.metric(label=name, value=f"{rate:+.1f}%", delta=f"{diff:.3f}")
                
                results_list.append({
                    "분석 지표": name.split(' ')[0],
                    "시청 전 평균": f"{pre_val:.4f}",
                    "시청 후 평균": f"{post_val:.4f}",
                    "증감률": f"{rate:+.2f}%"
                })

        st.divider()

        # 5. 시각화
        c1, c2 = st.columns([1, 1.5])
        with c1:
            st.write("### 📑 실시간 데이터 통계")
            st.table(pd.DataFrame(results_list))
        with c2:
            st.write("### 📈 뇌파 변화 추이 (시계열)")
            # 대표 채널 4종 시각화
            trend_cols = [c for c in ['Alpha_TP9', 'Beta_TP9', 'Theta_TP9', 'Delta_TP9'] if c in df.columns]
            if trend_cols:
                st.line_chart(df[trend_cols].tail(3000))

    except Exception as e:
        st.error(f"데이터 처리 중 오류 발생: {e}. CSV 파일 형식이 MUSE 2 표준인지 확인해 주세요.")
