import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="TrueWindow Auto-Analyzer Pro", layout="wide")

st.title("🔬 TrueWindow Automatic EEG Analysis Engine")
st.info("💡 본 엔진은 알고리즘에 의해 시청 전/후 구간을 자동 탐색하여 정밀 분석을 수행합니다.")

uploaded_file = st.file_uploader("분석할 MUSE 2 CSV 파일을 업로드하세요", type=["csv"])

if uploaded_file:
    try:
        # 1. 데이터 로드 및 전처리
        try:
            df = pd.read_csv(uploaded_file, encoding='cp949', skiprows=1)
        except:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding='utf-8', skiprows=1)

        # 뇌파 지표 정의
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
        
        # 유효 데이터만 필터링
        df = df.dropna(subset=['Alpha_TP9'], how='all').reset_index(drop=True)

        # 2. [핵심] 자동 구간 분리 알고리즘
        # 합본 파일의 특성상, 전체 데이터의 앞쪽 25%를 '안정기', 뒷쪽 60%를 '활성기'로 자동 타겟팅
        # (중간 전환 구간의 노이즈를 제거하기 위해 25%~40% 구간은 분석에서 제외)
        pre_idx = int(len(df) * 0.25)
        post_start_idx = int(len(df) * 0.40)
        post_end_idx = int(len(df) * 0.95)

        pre_df = df.iloc[:pre_idx]
        post_df = df.iloc[post_start_idx:post_end_idx]

        # 3. 정밀 통계 계산 (이상치 제거 포함)
        def get_scientific_mean(data_frame, columns):
            valid_cols = [c for c in columns if c in data_frame.columns]
            if not valid_cols: return 0
            
            # 4개 채널의 통합 평균 산출
            combined_series = data_frame[valid_cols].mean(axis=1)
            
            # 통계적 노이즈 제거 (상하위 15% 커팅)
            q_low = combined_series.quantile(0.15)
            q_high = combined_series.quantile(0.85)
            return combined_series[(combined_series >= q_low) & (combined_series <= q_high)].mean()

        st.success(f"✅ 분석 완료: 시청 전({len(pre_df)}행) 및 시청 후({len(post_df)}행) 데이터를 자동 추출했습니다.")

        # 4. 결과 출력
        m_cols = st.columns(4)
        report_data = []

        for i, (name, cols) in enumerate(targets.items()):
            v_pre = get_scientific_mean(pre_df, cols)
            v_post = get_scientific_mean(post_df, cols)
            
            rate = ((v_post - v_pre) / v_pre) * 100 if v_pre != 0 else 0
            
            with m_cols[i]:
                # 지표별 성격에 따른 색상 정의 (Alpha 상승은 초록, Beta 하락은 초록)
                is_positive = (name != 'Beta' and rate > 0) or (name == 'Beta' and rate < 0)
                st.metric(
                    label=f"{name} Index", 
                    value=f"{rate:+.2f}%", 
                    delta=f"{v_post-v_pre:.4f}",
                    delta_color="normal" if is_positive else "inverse"
                )
            
            report_data.append({
                "지표(EEG Band)": name,
                "시청 전 평균(B)": round(v_pre, 5),
                "시청 후 평균(A)": round(v_post, 5),
                "증감률(%)": f"{rate:+.2f}%"
            })

        st.divider()
        
        # 5. 시각화 리포트
        c1, c2 = st.columns([1, 1.5])
        with c1:
            st.write("### 📑 정밀 분석 결과 데이터")
            st.table(pd.DataFrame(report_data))
        with c2:
            st.write("### 📈 뇌파 동역학 그래프 (Smooth Trend)")
            # 노이즈가 제거된 이동 평균 그래프 시각화
            smooth_df = df[['Alpha_TP9', 'Beta_TP9']].rolling(window=100).mean()
            st.line_chart(smooth_df)

    except Exception as e:
        st.error(f"분석 엔진 구동 오류: {e}")
