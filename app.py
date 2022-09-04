import streamlit as st
import pandas as pd
import numpy as np
import joblib as jb


st.title('Thyroid Disorder Prediction')
# TSH,FTI,TT4,T3,query_hypothyroid,on_thyroxine,sex,pregnant,psych
TSH=st.number_input('TSH')
FTI=st.number_input('FTI')
TT4=st.number_input('TT4')
T3=st.number_input('T3')
query_hypothyroid=st.selectbox('query_hypothyroid',('True','False'))
on_thyroxine=st.selectbox('on_thyroxine',('True','False'))
sex=st.selectbox('sex',('Male','Female'))
pregnant=st.selectbox('pregnant',('True','False'))
psych=st.selectbox('psych',('True','False'))

button=st.button('predict')
if button==True:
    X=pd.DataFrame([[TSH,FTI,TT4,T3,query_hypothyroid,on_thyroxine,sex,pregnant,psych]],
                columns=['TSH','FTI','TT4','T3','query_hypothyroid','on_thyroxine','sex','pregnant','psych'])
    X['sex'].replace({'Male':1,'Female':0},inplace=True)
    X=X.replace(['True','False'],[1,0])
    model=jb.load('model_RandomForest.pkl')
    result=model.predict(X)
    if result==1:
        st.write('Your Values are Normal')
        st.image('thyroid.png')
    elif result==0:
        st.write('Your Values are Abnormal, Seems you have Thyroid')
    else:
        st.write('Your Values shows error while prediction')
    