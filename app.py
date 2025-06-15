
import streamlit as st
import pandas as pd
from prophet import Prophet
from datetime import datetime
import requests

st.set_page_config(page_title="환율 예측 AI", layout="wide")
st.title("💱 환율 예측 AI")

@st.cache_data
def fetch_exchange_rate():
    API_KEY = "99BO6UEVOS1ZHTSHK79J"
    start_date = "20240101"
    end_date = datetime.today().strftime("%Y%m%d")
    url = f"http://ecos.bok.or.kr/api/StatisticSearch/{API_KEY}/json/kr/1/1000/036Y001/DD/{start_date}/{end_date}/0002"

    try:
        response = requests.get(url)
        st.write("📡 응답 상태 코드:", response.status_code)
        st.json(response.json())
        data = response.json()
        rows = data['StatisticSearch']['row']
        df = pd.DataFrame(rows)
        df = df[['TIME', 'DATA_VALUE']]
        df.columns = ['ds', 'y']
        df['ds'] = pd.to_datetime(df['ds'])
        df['y'] = df['y'].astype(float)
        return df
    except Exception as e:
        st.error(f"환율 API 로딩 실패: {e}")
        return None

df = fetch_exchange_rate()

if df is None:
    st.stop()

st.sidebar.header("예측 설정")
days = st.sidebar.slider("예측할 일 수", min_value=3, max_value=30, value=7)

try:
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)
    result = forecast[['ds', 'yhat']].tail(days)
    result.columns = ['날짜', '예측 환율 (KRW/USD)']

    st.subheader("📈 예측 결과")
    st.line_chart(result.set_index("날짜"))
    st.dataframe(result)

except Exception as e:
    st.error(f"모델 예측 실패: {e}")
