import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
import plotly.io as pio
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima.arima import auto_arima

data = pd.read_csv('INR-USD.csv')

data = data.dropna()
data['Date'] = pd.to_datetime(data['Date'], format = '%Y-%m-%d')
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month


figure = px.line(data, 
                x="Date", 
                y="Close", 
                title="USD - INR Conversion Rate over the years")
figure.show()

growth = data.groupby('Year').agg({'Close': lambda x: (x.iloc[-1]-x.iloc[0])/x.iloc[0]*100})

fig = go.Figure()
fig.add_trace(go.Bar(x=growth.index,
                     y=growth['Close'],
                     name='Yearly Growth'))

fig.update_layout(title = 'Yearly Growth of USD - INR Conversion Rate',
                  xaxis_title = 'Year',
                  yaxis_title = 'Growth (%)',
                  width=900,
                  height=600)
pio.show(fig)

data['Growth'] = data.groupby(['Year', 'Month'])['Close'].transform(lambda x: (x.iloc[-1]-x.iloc[0])/x.iloc[0]*100)

grouped_data = data.groupby('Month').mean().reset_index()
fig = go.Figure()
fig.add_trace(go.Bar(x=grouped_data['Month'],
                     y=grouped_data['Growth'],
                     marker_color=grouped_data['Growth'],
                     hovertemplate='Month: %{x}<br>Average Growth: %{y:.2f}%<extra></extra>'))

fig.update_layout(title='Aggregated Monthly Growth of USD - INR Conversion Rate',
                  xaxis_title = 'Month',
                  yaxis_title = 'Average Growth (%)',
                  width=900,
                  height=600)
pio.show(fig)

result = seasonal_decompose(data['Close'], model='multiplicative', period=24)
fig = plt.figure()
fig = result.plot()
fig.set_size_inches(8,6)
plt.show()

model = auto_arima(data['Close'], seasonal = True, m=52, suppress_warnings=True)
p, d, q = 2, 1, 0

model = SARIMAX(data['Close'], order=(p, d, q), seasonal_order=(p, d, q, 52))

fitted = model.fit()
predictions = fitted.predict(len(data), len(data)+60)

fig = go.Figure()
fig.add_trace(go.Scatter(x=predictions.index,
              y=predictions,
              mode='lines',
              name='Predictions',
              line=dict(color='green')))

fig.update_layout(
    title='INR Rate - Training Data and Predictions',
    xaxis_title = 'Date',
    yaxis_title = 'Close',
    legend_title = 'Data',
    width=900,
    height=600
)

pio.show(fig)