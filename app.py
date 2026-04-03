import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="TrueWindow Neuro-AI Analyzer", layout="wide")

st.title("🪟 TrueWindow Real-Time Neuro-Analyzer")
st.info("💡 업로드된 CSV 파일의 원천 데이터를 실시간으로 전수 조사하여 분석 결과를 도출합니다.")

uploaded_file = st.file_uploader("분석할 CSV 파일을 업로드하세요", type=["csv"])

if uploaded_file:
    try:
        # 1. 데이터 로드 (첫 줄 타이틀 건너뛰고 두 번째 줄을 헤더로 인식)
        df = pd.read_csv(uploaded_file, skiprows=1)
        
        # 2. 분석에 필요한 핵심 컬럼 정의 (MUSE 2 표준)
        targets = {
            'Alpha': ['Alpha_TP9', 'Alpha_AF7', 'Alpha_AF8', 'Alpha_TP10'],
            'Beta': ['Beta_TP9', 'Beta_AF7', 'Beta_AF8', 'Beta_TP10'],
            'Theta': ['Theta_TP9', 'Theta_AF7', 'Theta_AF8', 'Theta_TP10'],
            'Delta': ['Delta_TP9', 'Delta_AF7', 'Delta_AF8', 'Delta_TP10']
        }

        # 3. 데이터 등분 (시청 전/후 비교를 위해 데이터를 반으로 나눔)
        # 실제 환경에서는 타임스탬프 기준으로 나누나, 합본 파일 특성상 절반 지점을 기준으로 계산
        mid_point = len(df) // 2
        pre_df = df.iloc[:mid_point]
        post_df = df.iloc[mid_point:]

        st.success(f"총 {len(df):,}개의 데이터 행을 성공적으로 분석했습니다.")

        # 4. 결과 계산 및 출력
        cols = st.columns(4)
        
        final_results = []
        for i, (name, col_list) in enumerate(targets.items()):
            # 존재하는 컬럼만 추출
            valid_cols = [c for c in col_list if c in df.columns]
            
            if valid_cols:
                pre_avg = pre_df[valid_cols].mean().mean()
                post_avg = post_df[valid_cols].mean().mean()
                change_rate = ((post_avg - pre_avg) / pre_avg) * 100
                
                with cols[i]:
                    st.metric(
                        label=f"{name}파 변화율", 
                        value=f"{change_rate:.1f}%", 
                        delta=f"{post_avg - pre_avg:.3f} (절대값)"
                    )
                
                final_results.append({
                    "지표": name,
                    "시청 전 평균": f"{pre_avg:.4f}",
                    "시청 후 평균": f"{post_avg:.4f}",
                    "변화율": f"{change_rate:+.1f}%"
                })

        st.divider()
        
        # 5. 상세 데이터 테이블 및 그래프
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.write("### 📑 데이터 기반 상세 분석표")
            st.table(pd.DataFrame(final_results))
            
        with col_right:
            st.write("### 📈 뇌파 흐름 시각화 (Raw Trend)")
            # 각 지표의 대표 채널 하나씩 시각화
            plot_cols = [c for c in ['Alpha_TP9', 'Beta_TP9', 'Theta_TP9', 'Delta_TP9'] if c in df.columns]
            st.line_chart(df[plot_cols].tail(2000))

    except Exception as e:
        st.error(f"데이터 처리 중 오류 발생: {e}. CSV 파일의 형식을 확인해 주세요.")
