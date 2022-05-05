# -*- coding: utf-8 -*-


from numpy import empty
from pandas.core.frame import DataFrame
import streamlit as st
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import bist


st.header("BIST30")
st.image("stocks.jpg")

st.subheader("Keşifsel Veri analizi")
st.write("---")
st.sidebar.title("Stock Symbols")
 
#stockname= tuple(pd.read_csv('stocks.csv')["Name"].to_list())
symbols = tuple(pd.read_csv('stocks.csv')["Symbol"].to_list())
symbol = st.sidebar.selectbox(label="Select stock symbol", options=symbols)



if symbol:
    df = bist.runAll(symbol)
    st.write(df.sort_values(by="Market_date",ascending=False))
    fig = go.Figure(data=[go.Candlestick(x=df['Market_date'], open=df['Open Price ₺'], high=df['High Price ₺'], low=df['Low Price ₺'], close=df['Close Price ₺'])])
    st.plotly_chart(fig)




st.subheader("Fiyat Tahmini")
st.write("---")

with st.form('my_form'):
    input_open = st.number_input("Açılış Fiyatı")
    input_high = st.number_input("En Yüksek Fiyat")
    input_low = st.number_input("En Düşük Fiyat")
    input_vol = st.number_input("Hacim")
    predict_button = st.form_submit_button("Tahmin Et")

    if predict_button:
        prediction = bist.train_and_predict(df, input_open, input_high,  input_low, input_vol)
        st.subheader(f'{prediction[0]:.3f}')
