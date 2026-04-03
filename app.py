import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="TrueWindow Neuro-Analysis Engine", layout="wide")

st.title("🔬 TrueWindow Professional EEG Analyzer")
st.markdown("---")

uploaded_file = st.file_uploader("MUSE 2 CSV 파일을 업로드하세요", type=["csv"])

if uploaded_file:
    try:
        # 1. 데이터 로드 (첫 줄 타이틀 제외)
        try:
            df = pd.read_csv(uploaded_file, encoding='cp949', skiprows=1)
        except:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding='utf-8', skiprows=1)

        # 2. 전처리: 숫자가 아닌 데이터 제거 및 숫자 변환
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
        df = df.dropna(subset=['Alpha_TP9'], how='all')

        # 3. 사이드바 제어 (구간 설정)
        st.sidebar.header("🎯 분석 구간 정밀 설정")
        st.sidebar.info("시청 전/후의 정확한 안정기를 선택하세요.")
        total = len(df)
        pre_range = st.sidebar.slider("시청 전(Baseline) 구간", 0, 100, (5, 20))
        post_range = st.sidebar.slider("시청 중(TrueWindow) 구간", 0, 100, (45, 85))

        pre_data = df.iloc[int(total*pre_range[0]/100):int(total*pre_range[1]/100)]
        post_data = df.iloc[int(total*post_range[0]/100):int(total*post_range[1]/100)]

        # 4. 전문가용 통계 분석 로직 (이상치 제거)
        def get_clean_mean(data_frame, columns):
            valid_cols = [c for c in columns if c in data_frame.columns]
            if not valid_cols: return 0
            
            subset = data_frame[valid_cols].mean(axis=1)
            # IQR 기법으로 튀는 값 제거 (상하위 10% 제거)
            q1 = subset.quantile(0.1)
            q3 = subset.quantile(0.9)
            cleaned = subset[(subset >= q1) & (subset <= q3)]
            return cleaned.mean()

        if len(pre_data) > 0 and len(post_data) > 0:
            st.success(f"✅ 분석 완료: {len(pre_data)}행(전) vs {len(post_data)}행(후) 비교 중")
            
            m_cols = st.columns(4)
            final_report = []

            for i, (name, cols) in enumerate(targets.items()):
                v_pre = get_clean_mean(pre_data, cols)
                v_post = get_clean_mean(post_data, cols)
                
                # 변화율 계산
                rate = ((v_post - v_pre) / v_pre) * 100 if v_pre != 0 else 0
                
                with m_cols[i]:
                    color = "normal" if (name == 'Beta' and rate < 0) or (name != 'Beta' and rate > 0) else "inverse"
                    st.metric(label=f"{name} 지표", value=f"{rate:+.2f}%", delta=f"{v_post-v_pre:.4f}", delta_color=color)
                
                final_report.append({"지표": name, "시청전": round(v_pre, 4), "시청후": round(v_post, 4), "증감률": f"{rate:+.2f}%"})

            st.divider()
            c1, c2 = st.columns([1, 2])
            with c1:
                st.write("### 📑 상세 분석표")
                st.table(pd.DataFrame(final_report))
            with c2:
                st.write("### 📈 전체 시계열 트렌드")
                # Alpha파 기준 시각화
                st.line_chart(df[['Alpha_TP9', 'Alpha_AF7']].rolling(window=50).mean())

    except Exception as e:
        st.error(f"엔진 오류: {e}")
