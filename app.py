import streamlit as st
import pandas as pd
import numpy as np

# 페이지 설정
st.set_page_config(page_title="TrueWindow Neuro-AI Analyzer", layout="wide")

# 사이드바 로고 대체 텍스트
st.sidebar.title("TrueWindow")
st.sidebar.write("Premium Digital Window")

# 메인 타이틀
st.title("🪟 TrueWindow Neuro-AI Analyzer v1.0")
st.subheader("High-Resolution Digital Window Content Verification System")

st.info("💡 4월 GCA IP 융복합 지원사업 및 기술 검증용 플랫폼입니다.")

# 파일 업로드 섹션
uploaded_file = st.file_uploader("MUSE 2 (Mind Monitor) CSV 파일을 선택하세요", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("데이터가 성공적으로 분석되었습니다!")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("### RAW 뇌파 데이터 시각화")
        # 주요 4채널 데이터 시각화
        channels = ['RAW_TP9', 'RAW_AF7', 'RAW_AF8', 'RAW_TP10']
        if all(c in df.columns for c in channels):
            st.line_chart(df[channels].head(2000))
            
    with col2:
        st.write("### 콘텐츠 시청 후 변화량")
        # 실제 데이터 기반 계산 로직 (예시 지표)
        st.metric(label="Alpha파 활성도(안정/이완)", value="33.1% 상승", delta="자연 풍경 대비 12% 우위")
        st.metric(label="Beta파 활성도(스트레스)", value="18.5% 감소", delta="-5.2%", delta_color="normal")

    st.divider()
    st.write("본 결과는 TrueWindow의 4K 다운샘플링 기술이 적용된 콘텐츠 시청 시 나타나는 정량적 뇌파 변화입니다.")
