import streamlit as st
import pandas as pd
from prophet import Prophet
from datetime import datetime, timedelta

st.set_page_config(page_title="환율 예측 AI", layout="wide")
st.title("💱 환율 예측 AI 시스템")

# 사용자 입력
mode = st.radio("예측 방식", ["Prophet 기반 예측", "시연용 더미 데이터"])
start_date = st.date_input("예측 시작 날짜", datetime.today())
days = st.slider("예측 일 수", min_value=1, max_value=30, value=7)

if mode == "Prophet 기반 예측":
    try:
        df = pd.read_csv("data/exchange_rate.csv")
        df.columns = ['ds', 'y']  # Prophet 요구 포맷
        model = Prophet()
        model.fit(df)

        future = model.make_future_dataframe(periods=days)
        forecast = model.predict(future)
        result = forecast[['ds', 'yhat']].tail(days)
        result.columns = ['날짜', '예측 환율 (KRW/USD)']
    except Exception as e:
        st.error(f"예측 실패: {e}")
        st.stop()
else:
    import numpy as np
    dates = [start_date + timedelta(days=i) for i in range(days)]
    rates = np.random.normal(1300, 5, size=days)
    result = pd.DataFrame({
        "날짜": dates,
        "예측 환율 (KRW/USD)": rates
    })

# 출력
st.line_chart(result.set_index("날짜"))
st.dataframe(result)
