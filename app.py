import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="TrueWindow Neuro-Analysis Pro", layout="wide")

st.title("🔬 TrueWindow Neuro-AI Analysis Platform (Pro)")
st.markdown("---")

uploaded_file = st.file_uploader("분석할 MUSE 2 CSV 파일을 업로드하세요", type=["csv"])

if uploaded_file:
    try:
        # 1. 데이터 로드 및 인코딩 해결
        try:
            df = pd.read_csv(uploaded_file, encoding='cp949', skiprows=1)
        except:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding='utf-8', skiprows=1)

        # 2. 데이터 정제 (문자열 제거 및 숫자 변환)
        targets = {
            'Alpha': ['Alpha_TP9', 'Alpha_AF7', 'Alpha_AF8', 'Alpha_TP10'],
            'Beta': ['Beta_TP9', 'Beta_AF7', 'Beta_AF8', 'Beta_TP10'],
            'Theta': ['Theta_TP9', 'Theta_AF7', 'Theta_AF8', 'Theta_TP10'],
            'Delta': ['Delta_TP9', 'Delta_AF7', 'Delta_AF8', 'Delta_TP10']
        }
        
        all_cols = [c for sub in targets.values() for c in sub]
        for col in all_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna(subset=[c for c in all_cols if c in df.columns], how='all')

        # 3. 구간 선택 사이드바 (핵심 기능)
        st.sidebar.header("⚙️ 분석 구간 설정")
        total_rows = len(df)
        split_range = st.sidebar.slider(
            "시청 전(Baseline) 구간 범위 (%)", 
            0, 100, (0, 20)
        )
        test_range = st.sidebar.slider(
            "시청 중(TrueWindow) 구간 범위 (%)", 
            0, 100, (30, 90)
        )

        # 데이터 슬라이싱
        pre_df = df.iloc[int(total_rows*(split_range[0]/100)):int(total_rows*(split_range[1]/100))]
        post_df = df.iloc[int(total_rows*(test_range[0]/100)):int(total_rows*(test_range[1]/100))]

        if len(pre_df) > 0 and len(post_df) > 0:
            st.success(f"✅ 분석 구간 설정 완료: Baseline({len(pre_df)}행) vs Test({len(post_df)}행)")
            
            # 4. 결과 계산 및 메트릭 출력
            m_cols = st.columns(4)
            analysis_results = []

            for i, (name, col_list) in enumerate(targets.items()):
                valid_cols = [c for c in col_list if c in df.columns]
                if valid_cols:
                    # 이상치 제거 (상하위 5% 커팅으로 정밀도 향상)
                    def clean_mean(data):
                        q_low = data.quantile(0.05)
                        q_hi  = data.quantile(0.95)
                        return data[(data > q_low) & (data < q_hi)].mean()

                    v_pre = pre_df[valid_cols].apply(clean_mean).mean()
                    v_post = post_df[valid_cols].apply(clean_mean).mean()
                    
                    rate = ((v_post - v_pre) / v_pre) * 100 if v_pre != 0 else 0
                    
                    with m_cols[i]:
                        st.metric(label=f"{name} 변화율", value=f"{rate:+.1f}%", delta=f"{v_post-v_pre:.3f}")
                    
                    analysis_results.append({
                        "지표": name,
                        "시청 전 평균": round(v_pre, 4),
                        "시청 후 평균": round(v_post, 4),
                        "증감률": f"{rate:+.2f}%"
                    })

            # 5. 시각화 및 리포트
            st.divider()
            c1, c2 = st.columns([1, 2])
            with c1:
                st.write("### 📑 상세 분석 리포트")
                st.table(pd.DataFrame(analysis_results))
            with c2:
                st.write("### 📈 구간별 뇌파 흐름 (Alpha파 기준)")
                st.line_chart(df[['Alpha_TP9', 'Alpha_AF7']].iloc[int(total_rows*0.1):int(total_rows*0.9)])

        else:
            st.warning("측정 구간을 설정해 주세요. 슬라이더를 조절하여 분석 범위를 지정할 수 있습니다.")

    except Exception as e:
        st.error(f"분석 엔진 오류: {e}")
