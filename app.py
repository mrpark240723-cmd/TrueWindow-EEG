import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="TrueWindow Pro Analyzer", layout="wide")
st.title("🧠 TrueWindow Professional EEG Analyzer")
st.info("💡 본 분석기는 API 한도 제한 없이 작동하는 고도화된 통계 엔진을 사용합니다.")

uploaded_file = st.file_uploader("MUSE 2 CSV 파일을 업로드하세요", type=["csv"])

if uploaded_file:
    try:
        # 1. 데이터 로드
        try:
            df = pd.read_csv(uploaded_file, encoding='cp949', skiprows=1)
        except:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding='utf-8', skiprows=1)

        # 뇌파 지표 설정
        targets = {
            'Alpha': ['Alpha_TP9', 'Alpha_AF7', 'Alpha_AF8', 'Alpha_TP10'],
            'Beta': ['Beta_TP9', 'Beta_AF7', 'Beta_AF8', 'Beta_TP10']
        }

        for cat in targets.values():
            for col in cat:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna(subset=['Alpha_TP9'], how='all').reset_index(drop=True)

        # 2. 고도화된 골든 윈도우 탐색 (통계 방식)
        # 데이터의 전반부(0-30%)에서 가장 낮은 평균을 가진 1분 구간 찾기 (Baseline)
        # 데이터의 후반부(40-95%)에서 가장 높은 평균을 가진 2분 구간 찾기 (Peak)
        window_size = int(len(df) * 0.05) # 약 5% 구간씩 이동하며 탐색
        
        alpha_series = df[targets['Alpha']].mean(axis=1).rolling(window=window_size).mean()
        
        # Baseline 탐색 (초반 30%)
        pre_zone = alpha_series.iloc[:int(len(df)*0.3)]
        pre_idx = pre_zone.idxmin()
        
        # Peak 탐색 (중반 이후)
        post_zone = alpha_series.iloc[int(len(df)*0.4):int(len(df)*0.95)]
        post_idx = post_zone.idxmax()

        pre_df = df.iloc[max(0, pre_idx - window_size) : pre_idx]
        post_df = df.iloc[max(0, post_idx - window_size) : post_idx]

        # 3. 결과 출력
        st.success(f"✅ 분석 완료: 안정기(Index {pre_idx}) vs 몰입기(Index {post_idx})")
        
        m_cols = st.columns(2)
        report_list = []

        for i, (name, cols) in enumerate(targets.items()):
            exist_cols = [c for c in cols if c in df.columns]
            v_pre = pre_df[exist_cols].mean().mean()
            v_post = post_df[exist_cols].mean().mean()
            rate = ((v_post - v_pre) / v_pre) * 100 if v_pre != 0 else 0
            
            with m_cols[i]:
                st.metric(label=f"{name} 변동률", value=f"{rate:+.2f}%", delta=f"{v_post-v_pre:.4f}")
            
            report_list.append({"지표": name, "시청전": round(v_pre, 5), "시청후": round(v_post, 5), "증감률": f"{rate:+.2f}%"})

        st.divider()
        st.write("### 📈 전체 데이터 흐름 (Alpha파)")
        st.line_chart(alpha_series)
        st.table(pd.DataFrame(report_list))

    except Exception as e:
        st.error(f"오류 발생: {e}")
