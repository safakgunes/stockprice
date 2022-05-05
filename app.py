# -*- coding: utf-8 -*-
from numpy import empty
from pandas.core.frame import DataFrame
import streamlit as st
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
import bist
import lstm

my_page = st.sidebar.radio('Page Navigation', ['Teknik Analiz ve Tahmin', 'Anomali Tespiti'])
symbols = tuple(pd.read_csv('stocks.csv')["Symbol"].to_list())
st.sidebar.title("Hisse Senetleri")
symbol = st.sidebar.selectbox(label="Bir Hisse Senedi Seçiniz", options=symbols)
scaler = StandardScaler()


if my_page == 'Teknik Analiz ve Tahmin':


    st.header("BIST30")
    st.image("stocks.jpg")

    st.subheader("Keşifsel Veri analizi")
    st.write("---")
    
    #stockname= tuple(pd.read_csv('stocks.csv')["Name"].to_list())
   



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


    


else:
    st.title('Anomali Tespiti')
    if symbol:
    

       
        test_score_df,anomalies,inverse_test,inverse_anomaly = lstm.lstm(symbol)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=test_score_df['Market_date'], y=inverse_test, name='Close price'))
        fig.add_trace(go.Scatter(x=anomalies['Market_date'], y=inverse_anomaly, mode='markers', name='Anomaly'))
        fig.update_layout(showlegend=True, title='Detected anomalies')
        st.plotly_chart(fig)
        
       


       


       

   
