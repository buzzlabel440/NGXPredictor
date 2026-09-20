import yfinance as yf
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error


def download_from_source(symbol, start_date, end_date):
    return yf.download(
        symbol,
        start=start_date,
        end=end_date,
        interval='1d'
    )

symbol = 'INFY.NS'
start_date= '2022-01-01'
end_date= '2023-01-01'

data  = download_from_source(symbol, start_date, end_date)
data = data.reset_index()
data.columns = data.columns.get_level_values(0)

data.to_csv('data/raw/stock_data.csv')
print(data.head())
print(data.columns)

def validate_data(df):
    print(f'Shape: {df.shape}')
    print(f"\nData range: {df.index.min()} to {df.index.max()}")
    print(f"\nMissing Values: \n{df.isnull().sum()}")
    print(f"\nData types: \n{df.dtypes}")
    print(f"\nClose Price Stat: \n{df['Close'].describe()}")
    return df

validate_data(data)


#psuedo structure
df = pd.read_csv('data/raw/stock_data.csv', parse_dates= ['Date'])
df = df.sort_values('Date')

x = df[['Close', 'High', 'Low', 'Volume']].shift(1).dropna()
y = df[['Close']].shift(-1).dropna()

# train/test split 80/20
x_train, x_test, y_train, y_test= train_test_split(
    x, y, test_size=0.2, shuffle=False #no shuffle preserve time order
)

model = LinearRegression()
model.fit(x_train, y_train)

predictions = model.predict(x_test)

# metrics 
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
mse = mean_squared_error(y_test, predictions)

print(f"MAE: ${mae:.2f}")
print(f"R2: {r2:.4f}")
print(f"MSE: {mse:.2f}")

corr = df['Close'].corr(df['Close'].shift(1))
print(f"autocorrelation Price Today vs tomorrow: {corr:.4f}")