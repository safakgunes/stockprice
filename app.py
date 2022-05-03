# -*- coding: utf-8 -*-


from numpy import empty
from pandas.core.frame import DataFrame
import streamlit as st
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import bist

st.header("BIST30 - ŞAFAK GÜNEŞ")
st.image("stocks.jpg")
st.subheader("Keşifsel Veri analizi")
st.write("---")
st.sidebar.title("Stock Symbols")

symbols = tuple(pd.read_csv('stocks.csv')["Symbol"].to_list())
symbol = st.sidebar.selectbox(label="Select stock symbol", options=symbols)

    
if symbol:
    df = bist.runAll(symbol)
    st.write(df)
    fig = go.Figure(data=[go.Candlestick(x=df['Market_date'], open=df['Open Price ₺'], high=df['High Price ₺'], low=df['Low Price ₺'], close=df['Close Price ₺'])])
    st.plotly_chart(fig)

# with st.form('Indicator_form'):
#     Indicator = st.selectbox(label="Select stock symbol", options=('C','RSI','BB','20d_MA'))
#     Indicator_Type = st.form_submit_button("Indicator")

#    if Indicator_Type:
#        st.write(Indicator)
#        if Indicator =="RSI":
#            st.write("RRR")
#            fig = px.line(df, x='Market_date', y='RSI', title='RSI Indicator')
#            st.plotly_chart(fig)
#        elif Indicator =="BB":
        #     st.write("BBB")
        # elif Indicator =="MA":
        #     st.write("MMM")
        # else:
        #     st.write("CCC")
st.subheader("Teknik İndikatörler")
st.write("---")
fig = px.line(df, x='Market_date', y=["RSI","20d_MA", "bb_bbh", "bb_bbl"])
#title='Indicator'
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
