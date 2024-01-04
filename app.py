import streamlit as st
import pandas as pd
import numpy as np
import math
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import plotly.express as px
from datetime import date
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score

import yfinance as yf

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Cryptocurrency Prediction App')

maindf = ('GOOG', 'AAPL', 'MSFT', 'GME')
selected_stock = st.selectbox('Select dataset for prediction', maindf)

@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

st.subheader('Raw data')
st.write(data.tail())

# Dropdown to select the type of price
selected_price_type = st.selectbox('Select price type', ['Close', 'High', 'Open', 'Low'])

fig_original = px.line(data, x='Date', y=selected_price_type, labels={'Date': 'Date', selected_price_type: f'{selected_price_type} Price'})
st.plotly_chart(fig_original)

# For Evaluation we will use these library
st.sidebar.header('Evaluation Metrics')
show_metrics = st.sidebar.checkbox('Show Evaluation Metrics', value=True)
if show_metrics:
    st.sidebar.subheader('Evaluation Metrics Options')
    selected_metrics = st.sidebar.multiselect('Select Metrics', ['Mean Squared Error', 'Mean Absolute Error', 'Explained Variance Score', 'R2 Score'], default=['Mean Squared Error'])

# For model building we will use these library
st.sidebar.header('Model Building Options')
selected_model_options = st.sidebar.checkbox('Show Model Building Options', value=True)
if selected_model_options:
    st.sidebar.subheader('Model Building Parameters')
    time_step = st.sidebar.slider('Time Step', min_value=1, max_value=30, value=15)
    epochs = st.sidebar.slider('Epochs', min_value=1, max_value=500, value=200)
    batch_size = st.sidebar.slider('Batch Size', min_value=1, max_value=128, value=32)

# Plotting the selected price type
st.write(f'Plotting {selected_price_type} Price:')
fig = px.line(data, x='Date', y=selected_price_type, labels={'Date': 'Date', selected_price_type: f'{selected_price_type} Stock'})
fig.update_traces(marker_line_width=2, opacity=0.8, marker_line_color='orange')
fig.update_layout(title_text=f'Bitcoin {selected_price_type} Price 2014-2024',
                  plot_bgcolor='white', font_size=15, font_color='black')
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
st.plotly_chart(fig)

# Extracting and normalizing the selected price type
price_df = data[['Date', selected_price_type]]
del price_df['Date']
scaler = MinMaxScaler(feature_range=(0, 1))
price_df = scaler.fit_transform(np.array(price_df).reshape(-1, 1))

# Data split for training and testing
training_size = int(len(price_df) * 0.60)
test_size = len(price_df) - training_size
train_data, test_data = price_df[0:training_size, :], price_df[training_size:len(price_df), :1]

def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step):
        a = dataset[i:(i + time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

# Creating datasets for LSTM
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Model Selection
model_type = st.sidebar.selectbox('Select Model Type', ['LSTM', 'FNN', 'MLP'])

if model_type == 'LSTM':
    model = Sequential()
    model.add(LSTM(10, input_shape=(X_train.shape[1], X_train.shape[2]), activation="relu"))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error", optimizer="adam")
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, verbose=1)

elif model_type == 'FNN' or model_type == 'MLP':
    model = Sequential()
    model.add(Dense(50, input_dim=time_step, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    history = model.fit(X_train.reshape(X_train.shape[0], -1), y_train, validation_data=(X_test.reshape(X_test.shape[0], -1), y_test),
                        epochs=epochs, batch_size=batch_size, verbose=1)

# Plot Training and Validation Loss
st.write('Training and Validation Loss:')
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(loss))
fig, ax = plt.subplots()
ax.plot(epochs, loss, 'r', label='Training loss')
ax.plot(epochs, val_loss, 'b', label='Validation loss')
ax.set_title('Training and validation loss')
ax.legend(loc=0)
st.pyplot(fig)

# Model Evaluation Metrics
if show_metrics:
    st.write('Model Evaluation Metrics:')
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    # Transform back to the original form
    train_predict = scaler.inverse_transform(train_predict.reshape(-1, 1))
    test_predict = scaler.inverse_transform(test_predict.reshape(-1, 1))
    original_ytrain = scaler.inverse_transform(y_train.reshape(-1, 1))
    original_ytest = scaler.inverse_transform(y_test.reshape(-1, 1))

    if 'Mean Squared Error' in selected_metrics:
        st.write("Train data Mean Squared Error:", mean_squared_error(original_ytrain, train_predict))
        st.write("Test data Mean Squared Error:", mean_squared_error(original_ytest, test_predict))
    if 'Mean Absolute Error' in selected_metrics:
        st.write("Train data Mean Absolute Error:", mean_absolute_error(original_ytrain, train_predict))
        st.write("Test data Mean Absolute Error:", mean_absolute_error(original_ytest, test_predict))
    if 'Explained Variance Score' in selected_metrics:
        st.write("Train data Explained Variance Score:", explained_variance_score(original_ytrain, train_predict))
        st.write("Test data Explained Variance Score:", explained_variance_score(original_ytest, test_predict))
    if 'R2 Score' in selected_metrics:
        st.write("Train data R2 Score:", r2_score(original_ytrain, train_predict))
        st.write("Test data R2 Score:", r2_score(original_ytest, test_predict))

# Plot Predicted vs Actual selected price type
st.write(f'Comparing Original vs Predicted {selected_price_type} Price:')
trainPredictPlot = np.empty_like(price_df)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[time_step:len(train_predict) + time_step, :] = train_predict

testPredictPlot = np.empty_like(price_df)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict) + (time_step * 2) + 1:len(price_df), :] = test_predict[:len(price_df) - len(train_predict) - (time_step * 2) - 1, :]

plotdf = pd.DataFrame({'date': data['Date'],
                       f'original_{selected_price_type.lower()}': scaler.inverse_transform(price_df).reshape(1, -1)[0].tolist(),
                       f'train_predicted_{selected_price_type.lower()}': trainPredictPlot.reshape(1, -1)[0].tolist(),
                       f'test_predicted_{selected_price_type.lower()}': testPredictPlot.reshape(1, -1)[0].tolist()})

# Define names for traces
trace_names = iter(['Original', 'Train Predicted', 'Test Predicted'])

fig_price_type = px.line(plotdf, x=plotdf['date'],
                         y=[plotdf[f'original_{selected_price_type.lower()}'],
                            plotdf[f'train_predicted_{selected_price_type.lower()}'],
                            plotdf[f'test_predicted_{selected_price_type.lower()}']],
                         labels={'value': f'{selected_price_type} price', 'date': 'Date'})

# Assign names to traces
fig_price_type.for_each_trace(lambda t: t.update(name=next(trace_names)))

fig_price_type.update_layout(title_text=f'Comparison between original {selected_price_type.lower()} vs predicted {selected_price_type.lower()}',
                             plot_bgcolor='white', font_size=15, font_color='black', legend_title_text=f'{selected_price_type} Price')
fig_price_type.update_xaxes(showgrid=False)
fig_price_type.update_yaxes(showgrid=False)
st.plotly_chart(fig_price_type)

# Predicting the next days
pred_days = st.slider('Select the number of days to predict', 10, 30, 30)

x_input = test_data[len(test_data)-time_step:].reshape(1, -1)
temp_input = list(x_input)
temp_input = temp_input[0].tolist()

lst_output = []
n_steps = time_step
i = 0
while i < pred_days:
    if len(temp_input) > time_step:
        x_input = np.array(temp_input[1:])
        x_input = x_input.reshape(1, -1)
        if model_type == 'LSTM':
            x_input = x_input.reshape((1, n_steps, 1))
        yhat = model.predict(x_input, verbose=0)
        temp_input.extend(yhat[0].tolist())
        temp_input = temp_input[1:]
        lst_output.extend(yhat.tolist())
        i += 1
    else:
        x_input = x_input.reshape((1, n_steps, 1)) if model_type == 'LSTM' else x_input.reshape(1, -1)
        yhat = model.predict(x_input, verbose=0)
        temp_input.extend(yhat[0].tolist())
        i += 1

# Creating the DataFrame for visualization
next_predicted_days_value = scaler.inverse_transform(np.array(lst_output).reshape(-1, 1)).reshape(1, -1).tolist()[0]

# Ensure the length is the same as the predicted days
next_predicted_days_value = next_predicted_days_value[:pred_days]

new_pred_plot = pd.DataFrame({
    'next_predicted_days_value': next_predicted_days_value
})

# Plotting the results
st.write(f'Predicted Prices for the Next {pred_days} days using {model_type}:')
fig = px.line(new_pred_plot, x=new_pred_plot.index, y=new_pred_plot['next_predicted_days_value'],
              labels={'value': f'{selected_price_type} price', 'index': 'Timestamp'})
fig.update_layout(title_text=f'Predicted Prices for the Next {pred_days} days using {model_type}',
                  plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='Close Price')

fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
st.plotly_chart(fig)
