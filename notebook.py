import datetime
from datetime import date
import qgrid
from IPython.core import debugger
from IPython.display import display, HTML
from ipywidgets import interact, widgets, Layout, interactive
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import plotly.graph_objects as go

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.layers import Dense, Dropout, LSTM
import keras
import pandas_datareader.data as web
import keras.metrics as metrics
import ipywidgets as widgets





# SET VARIABLE INPUTS 


start_time = widgets.DatePicker(description = 'Pick a start date:', disabled = False, value=datetime.date.today() - datetime.timedelta(days=1), layout=Layout(align_items='center'))

ticker = widgets.Text(value='Insert a valid stock ticker', placeholder='Insert a valid stock ticker',
                      description = 'Ticker:', disabled=False)

num_epochs = widgets.Dropdown(options=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], value=1, description='Number of epochs', disabled=False) 

select_button = widgets.Button(description='Go', disabled=False, button_style='primary',layout=Layout(margin=('0px 0px 0px 10px'), width = '10%'))

# end_time = widgets.DatePicker(description='Pick an end date', disabled = False, button_style='primary', layout=Layout(margin=('0px 0px 0px 10px'), width = '10%'))

data_output = widgets.Output()

print('1%')

# LOAD DATA 
def ai_load_data():
    h_box = widgets.HBox([start_time, ticker, num_epochs, select_button], layout=Layout(align_items='stretch'), margin='0px 0px 0px 0px')
 #   v_box = widgets.VBox([data_output], layout=Layout(align_items ='stretch'), margin=('0px 0px 0px 0px'))
    display(h_box)
 #   display(v_box)
    
    print('2%')
    # CALLABLE FUNCTION
    def automatic(a):
        
        print('3%')
        
        print(str(start_time.value))
        print(str(num_epochs.value))

        dataset = web.DataReader(ticker.value, data_source='yahoo',
                                 start = start_time.value, end=datetime.date.today())
        print(dataset.head(5))
        
        dataset = dataset.reset_index()
        
        dataset['Date'] = pd.to_datetime(dataset['Date'], format= '%Y-%m-%d')
        
        dataset.index = dataset['Date']
        
        dataset = dataset.sort_index(ascending = True, axis=0)
        
        dataset.drop(columns=['Open', 'High', 'Low', 'Volume'], inplace =True)

        print('3%')

        close_data = dataset['Close'].values
        
        close_data = close_data.reshape(-1,1)

        split_percent = 0.8
        
        split = int(split_percent * len(close_data))

        close_train = close_data[:split]
        
        close_test = close_data[split:]
        
        data_train = dataset['Date'][:split]
        
        date_test = dataset['Date'][split:] 


        look_back = 15
        train_generator = TimeseriesGenerator(close_train, close_train, length = look_back, batch_size = 20) 
        test_generator = TimeseriesGenerator(close_test, close_test, length = look_back, batch_size = 1)



        # MODEL CREATION
        model = Sequential()
 
        model.add(LSTM(500, activation = 'relu', input_shape=(look_back, 1), return_sequences = True))
        
        model.add(LSTM(500, activation = 'relu', input_shape=(look_back,1)))
        
        model.add(Dense(1))
        
        model.compile(optimizer ='adam', loss='mse', metrics=[metrics.MeanSquaredError(name='MSE'),
                                                     metrics.MeanAbsoluteError(name='Mean_Absolute_Error'),
                                                    metrics.RootMeanSquaredError(name='RMSE')])
        
        # EPOCHS AND TRAINING
        history = model.fit_generator(train_generator, epochs=num_epochs.value, verbose=1)
        
        Metrics = pd.DataFrame(history.history)
        del Metrics['loss']
        
        print(history.history.keys())
        
        plt.figure(figsize= (16, 8))
        
        plt.plot(history.history['loss'], color= 'black')
        
        plt.title('Cost Function Progression')
        
        plt.ylabel('Loss')
        
        plt.xlabel('epochs')
        
        print(list(Metrics.index[0:]))
        Metrics['epochs'] = list(Metrics.index[0:])
        print(Metrics.head(5))
     #   metrics_trace_1 = go.Scatter(x=Metrics['epochs'], y = Metrics['loss'])
        metrics_trace_1 = go.Scatter(x=Metrics['epochs'], y = Metrics['MSE'], mode = 'lines', name = 'Mean Squared Error')
        metrics_trace_2 = go.Scatter(x=Metrics['epochs'], y = Metrics['Mean_Absolute_Error'], mode = 'lines', name = 'Mean Absolute Error')
        metrics_trace_3 = go.Scatter(x=Metrics['epochs'], y = Metrics['RMSE'], mode = 'lines', name = 'Root Mean Squared Error')
                                     
        layout_hist = go.Layout(title = 'Loss Metrics across epochs', xaxis= {'title' : 'Epochs'}, yaxis = {'title': 'Value'})
        
        fig_hist = go.Figure(data=[metrics_trace_1, metrics_trace_2, metrics_trace_3], layout = layout_hist)
        
        fig_hist.show()

        
        print(Metrics)

        # PRELIMINARY PREDICTIONS
        prediction = model.predict_generator(test_generator)
        
        close_train = close_train.reshape((-1))
        
        close_test = close_test.reshape((-1))
        
        prediction = prediction.reshape((-1))

        # TRAINING PLOT
        trace1 = go.Scatter(x = data_train, y = close_train, mode = 'lines', name = 'Data') 
        
        trace2 = go.Scatter(x = date_test, y = prediction, mode = 'lines', name = 'Prediction')
        
        trace3 = go.Scatter(x = date_test, y = close_test, mode = 'lines', name = 'Truth')
        
        layout = go.Layout(title = str(ticker.value)+ ' ' + 'Stock', xaxis = {'title' : 'Date'}, yaxis = {'title': 'Close'})
        
        fig = go.Figure(data=[trace1, trace2, trace3], layout=layout)
        
        fig.show()

        close_data = close_data.reshape((-1))

        # FUNCTION DEF
        def predict(num_prediction, model):
            prediction_list = close_data[-look_back:]

            for i in range(num_prediction):
               
                x = prediction_list[-look_back:]
                
                x = x.reshape((1, look_back, 1 ))
                
                out = model.predict(x)[0][0]
                
                prediction_list = np.append(prediction_list, out)
            
            prediction_list = prediction_list[look_back -1:]
            
            return prediction_list

        def predict_dates(num_prediction):
            
            last_date = dataset['Date'].values[-1]
            
            prediction_dates = pd.date_range(last_date, periods=num_prediction+1).tolist()
            
            return prediction_dates

        # PREDICTION N.
        num_prediction = 30
        
        forecast = predict(num_prediction, model)
        
        forecast_dates = predict_dates(num_prediction)

        # PREDICTION PLOT
        predict_df = pd.DataFrame(forecast, forecast_dates)
        
        predict_df = predict_df.rename(columns={0:'Predicted Value'})
        
        qg_predict_df = qgrid.show_grid(predict_df)
        
        predict_trace = go.Scatter(x = predict_df.index, y = predict_df['Predicted Value'], mode = 'lines', name = 'Predicte value')
        
        layout_pred = go.Layout(title = str(ticker.value)+ ' ' + '30 Day Prediction', xaxis = {'title' : 'Date'}, yaxis = {'title': 'Closing Price'})
        
        fig_pred = go.Figure(data=predict_trace, layout=layout_pred)
        
        fig_pred.show()
        
 #       plt.figure(figsize=(16,8))
        
 #       predict_df.plot(color = 'black')
        
 #       plt.xlabel('Date')
        
 #       plt.ylabel('Closing Price')
        
 #       plt.title('30 Days Dynamic LSTM prediction ' + str(ticker.value))
        
 #       plt.show()
        
        print(predict_df)
        
    select_button.on_click(automatic)
    
    
    
                                                      
