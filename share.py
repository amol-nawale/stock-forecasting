import pandas as pd
import streamlit as st
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from keras.layers import Dropout
import tensorflow as tf
from sklearn.metrics import mean_squared_error,mean_absolute_percentage_error


sidebar=st.sidebar
sidebar.header('Input Space')

ticker=sidebar.text_input('Enter The Ticker Symbol')
start_date=sidebar.date_input('Enter The Starting Date')
end_date=sidebar.date_input('Enter The Ending Date')
button=sidebar.button('Submit')

if button:

    data=yf.Ticker(ticker)
    data_hist=data.history(start=start_date,end=end_date)

    data_hist.tail()

    def sequence(dataset,time_step=1):     
      x,y=[],[]
      for i in range(len(dataset)-time_step):
        a=dataset[i:(i+time_step)]
        x.append(a)
        y.append(dataset[i+time_step])
        return np.array(x),np.array(y)

    model_error_dict={}
    data=data_hist['Close']
    scaler=MinMaxScaler(feature_range=(0,1))
    df1=scaler.fit_transform(np.array(data).reshape(-1,1))
    for i in range(8):     
        train_data=df1
        time_step=100
        x_train,y_train=sequence(train_data,time_step)
        x_train=x_train.reshape(x_train.shape[0],x_train.shape[1],1)
        model=Sequential()
        model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))

        model.add(LSTM(50 ,return_sequences=True))
    # model.add(LSTM(32,return_sequences=True))
    # model.add(LSTM(32,return_sequences=True))
    # model.add(LSTM(32,return_sequences=True))
    # model.add(LSTM(32,return_sequences=True))
    # model.add(LSTM(128,return_sequences=True))
    # model.add(LSTM(128,return_sequences=True))
    # model.add(LSTM(64,return_sequences=True))
    # # model.add(LSTM(6,return_sequences=True,input_shape=(100,1)))

        model.add(LSTM(6))
    # model.add(Dropout(0.5))
        model.add(Dense(6))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error',optimizer='adam')

        history=model.fit(x_train,y_train,epochs = 100, batch_size =16,verbose=1)

        y_train_predict=model.predict(x_train)
        y_train_predict=scaler.inverse_transform(y_train_predict)
        a=scaler.inverse_transform(y_train)
        a=pd.DataFrame(a)
        a['predict']=y_train_predict
        a.rename(columns={0:'actual_train'},inplace=True)
        mean_squared_error_share=mean_squared_error(a['actual_train'],a['predict'])
#     print('mean_squared_error:',mean_squared_error(a['actual_train'],a['predict']))
#     print('mean_absolute_percentage_error:',mean_absolute_percentage_error(a['actual_train'],a['predict']))
        model_error_dict[i+1]={'model':model, 'mean_squared_error' : mean_squared_error_share}
        print(model_error_dict)


        for j in model_error_dict.keys():     
            if j==1: 
                minimum=model_error_dict[j]['mean_squared_error']
                model_use=model_error_dict[j]['model']
                key=j
            elif minimum > model_error_dict[j]['mean_squared_error']:
                minimum=model_error_dict[j]['mean_squared_error']
                model_use=model_error_dict[j]['model']
                key=j  
            else:

                minimum=minimum
                print(minimum)
                print(model_use)
                print(j)


    model=model_use
    X_ =train_data[-100:]  # last available input sequence
    X_ = X_.reshape(1, 100, 1)

    Y_ = model.predict(X_).reshape(-1, 1)
    Y_ = scaler.inverse_transform(Y_)

    x_input=train_data[-100:].reshape(1,-1)
    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()

    lst_output=[]
    n_steps=100
    i=1
    while(i<=12 ):
    
        if(len(temp_input)>100):
        #print(temp_input)
            x_input=np.array(temp_input[1:])
        # print("{} day input {}".format(i,x_input))
            x_input=x_input.reshape(1,-1)
            x_input = x_input.reshape((1, n_steps, 1))
        #print(x_input)
            yhat = model.predict(x_input, verbose=0)
            print(yhat)
            temp_input.extend(yhat[0].tolist())
            temp_input=temp_input[1:]
        #print(temp_input)
            lst_output.extend(yhat.tolist())
            i=i+1
        else:
            x_input = x_input.reshape((1, n_steps,1))
            yhat = model.predict(x_input, verbose=0)
            print(yhat[0])
            temp_input.extend(yhat[0].tolist())
        # print(len(temp_input))
            lst_output.extend(yhat.tolist())
            i=i+1
    
    lst_output=scaler.inverse_transform(lst_output)
    future=pd.DataFrame(lst_output)
    future.columns=['future']
    



    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(future)
    ax.set_title("Next 15 Days Forecast")
    st.pyplot(fig)
    future
    st.write(model_error_dict)







