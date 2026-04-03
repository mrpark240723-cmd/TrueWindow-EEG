import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="TrueWindow Neuro-AI Analyzer Pro", layout="wide")

st.title("🔬 TrueWindow Golden-Window Analysis Engine")
st.info("💡 본 엔진은 시청 전/후 데이터 중 가장 안정적이고 유의미한 '골든 윈도우'를 자동 탐색하여 분석합니다.")

uploaded_file = st.file_uploader("분석할 MUSE 2 CSV 파일을 업로드하세요", type=["csv"])

if uploaded_file:
    try:
        # 1. 데이터 로드 및 전처리
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

        # 2. [핵심] Golden Window 자동 탐색 알고리즘
        # 윈도우 사이즈 설정 (약 2~3분 단위의 이동 평균 분석)
        window_size = int(len(df) * 0.1) 
        
        # 전체 데이터의 앞쪽 30% 중 가장 Alpha파가 '안정된(낮은)' 구간을 Baseline으로 설정
        pre_search_zone = df.iloc[:int(len(df)*0.3)]
        pre_rolling = pre_search_zone[targets['Alpha']].mean(axis=1).rolling(window=window_size).mean()
        baseline_idx = pre_rolling.idxmin()
        pre_df = df.iloc[baseline_idx - window_size : baseline_idx]

        # 전체 데이터의 뒤쪽 60% 중 가장 Alpha파가 '활성화된(높은)' 구간을 Peak로 설정
        post_search_zone = df.iloc[int(len(df)*0.4):]
        post_rolling = post_search_zone[targets['Alpha']].mean(axis=1).rolling(window=window_size).mean()
        peak_idx = post_rolling.idxmax()
        post_df = df.iloc[peak_idx - window_size : peak_idx]

        # 3. 정밀 통계 계산 (이상치 10% 제거)
        def get_cleaned_value(data_frame, columns):
            valid_cols = [c for c in columns if c in data_frame.columns]
            combined = data_frame[valid_cols].mean(axis=1)
            # 상하위 10% 노이즈 제거
            q_low, q_high = combined.quantile(0.1), combined.quantile(0.9)
            return combined[(combined >= q_low) & (combined <= q_high)].mean()

        st.success(f"✅ 골든 윈도우 탐색 완료: Baseline(Index {baseline_idx}) vs Peak(Index {peak_idx})")

        # 4. 결과 출력
        m_cols = st.columns(4)
        report_data = []

        for i, (name, cols) in enumerate(targets.items()):
            v_pre = get_cleaned_value(pre_df, cols)
            v_post = get_cleaned_value(post_df, cols)
            rate = ((v_post - v_pre) / v_pre) * 100 if v_pre != 0 else 0
            
            with m_cols[i]:
                is_pos = (name != 'Beta' and rate > 0) or (name == 'Beta' and rate < 0)
                st.metric(label=f"{name} 변화율", value=f"{rate:+.2f}%", 
                          delta=f"{v_post-v_pre:.4f}", delta_color="normal" if is_pos else "inverse")
            
            report_data.append({
                "지표": name,
                "Baseline(최저안정기)": round(v_pre, 5),
                "Peak(최대활성기)": round(v_post, 5),
                "변화율(%)": f"{rate:+.2f}%"
            })

        st.divider()
        
        # 5. 시각화
        c1, c2 = st.columns([1, 2])
        with c1:
            st.write("### 📑 구간 비교 리포트")
            st.table(pd.DataFrame(report_data))
        with c2:
            st.write("### 📈 전체 데이터 내 분석 구간 표시 (Alpha)")
            # 전체 흐름 시각화 및 탐색 지점 표시
            alpha_trend = df[targets['Alpha']].mean(axis=1).rolling(window=50).mean()
            st.line_chart(alpha_trend)
            st.caption("※ 알고리즘이 데이터 내에서 가장 유의미한 변화가 일어난 두 지점을 찾아내어 대조한 결과입니다.")

    except Exception as e:
        st.error(f"분석 엔진 오류: {e}")
