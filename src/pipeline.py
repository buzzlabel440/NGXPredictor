import yfinance as yf
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import matplotlib.pyplot as plt



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



# Naive model: predict tomorrow = today
y_pred_naive = x_test['Close'].values  # Just use today's close as prediction
y_test_actual = y_test.values

r2_naive = r2_score(y_test_actual, y_pred_naive)
mae_naive = mean_absolute_error(y_test_actual, y_pred_naive)

print(f"Naive Model R²: {r2_naive:.6f}")
print(f"Naive Model MAE: ${mae_naive:.2f}")
print(f"\nYour Model R²: {0.8228:.6f}")
print(f"Your Model MAE: $14.54")
print(f"\nR² Improvement: {(0.8228 - r2_naive)*100:.2f}%")
print(f"MAE Improvement: {(mae_naive - 14.54)/mae_naive*100:.2f}%")



residuals = y_test.values - predictions
mean_residual = np.mean(residuals)
std_residual = np.std(residuals)

print(f"Mean Residual: ${mean_residual:.2f}")
print(f"Std Dev of Residuals: ${std_residual:.2f}")
print(f"Min Residual: ${np.min(residuals):.2f}")
print(f"Max Residual: ${np.max(residuals):.2f}")

# Plot
# plt.figure(figsize=(12, 5))

# plt.subplot(1, 2, 1)
# plt.plot(y_test.values, label='Actual', linewidth=2)
# plt.plot(predictions, label='Predicted', linewidth=2, alpha=0.7)
# plt.legend()
# plt.title('Actual vs Predicted Close (Test Set)')
# plt.ylabel('Price ($)')
# plt.xlabel('Day')

# plt.subplot(1, 2, 2)
# plt.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
# plt.title('Distribution of Prediction Errors')
# plt.xlabel('Residual ($)')
# plt.ylabel('Frequency')
# plt.axvline(mean_residual, color='red', linestyle='--', label=f'Mean: ${mean_residual:.2f}')
# plt.legend()

# plt.tight_layout()
# plt.show()



# =============Random Forest============== 
from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(
    n_estimators= 100,
    max_depth= 10,
    random_state=42
)

rf_model.fit(x_train, y_train)
rf_model_ypred= rf_model.predict(x_test)


# metrics
r2_score_rf= r2_score(y_test, rf_model_ypred)
rf_mae = mean_absolute_error(y_test, rf_model_ypred)

print(f"R2 Score: {r2_score_rf:.6f}")
print(f"MEA_RF: {rf_mae:.2f}")

# feauture importance
importances= rf_model.feature_importances_
features= ["Close", "Low", "High", "Volume"]

for feat, imp in zip(features, importances):
    print(f"{feat}: {imp:.2f}")


import numpy as np
from talib import RSI  # or calculate manually

def engineer_features(df):
    """Create exogenous-adjacent features"""
    
    # 1. Daily volatility (20-day rolling)
    df['volatility_20d'] = df['Close'].pct_change().rolling(20).std()
    
    # 2. Intraday range as % of close
    df['intraday_range_pct'] = (df['High'] - df['Low']) / df['Close']
    
    # 3. Volume ratio (today vs 20-day avg)
    df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
    
    # 4. Daily returns (momentum)
    df['daily_return'] = df['Close'].pct_change()
    
    # 5. Price change (close - open) as % of open
    df['co_range_pct'] = (df['Close'] - df['Open']) / df['Open']
    
    # 6. RSI (Relative Strength Index) - captures overbought/oversold
    df['rsi_14'] = RSI(df['Close'], timeperiod=14)
    
    # Drop NaN rows from rolling calculations
    df = df.dropna()
    
    return df

df_engineered = engineer_features(df)

# New feature set
new_features = ['Close', 'High', 'Low', 'Volume', 
                'volatility_20d', 'intraday_range_pct', 
                'volume_ratio', 'daily_return', 'co_range_pct', 'rsi_14']

X_new = df_engineered[new_features].shift(1).dropna()
y = df_engineered['Close'].shift(-1).dropna()

# Align lengths
X_new = X_new[:len(y)]

# Split and train
X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(
    X_new, y, test_size=0.2, shuffle=False
)

model_new = LinearRegression()
model_new.fit(X_train_new, y_train_new)

y_pred_new = model_new.predict(X_test_new)

r2_new = r2_score(y_test_new, y_pred_new)
mae_new = mean_absolute_error(y_test_new, y_pred_new)

print(f"New R²: {r2_new:.6f}")
print(f"New MAE: ${mae_new:.2f}")





# What if we REMOVE lagged close price entirely?
features_no_price = ['volatility_20d', 'intraday_range_pct', 
                      'volume_ratio', 'daily_return', 'co_range_pct', 'rsi_14']

X_derived = df_engineered[features_no_price].shift(1).dropna()
y = df_engineered['Close'].shift(-1).dropna()

X_derived = X_derived[:len(y)]
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
    X_derived, y, test_size=0.2, shuffle=False
)

model_derived = LinearRegression()
model_derived.fit(X_train_d, y_train_d)

r2_derived = r2_score(y_test_d, model_derived.predict(X_test_d))
print(f"R² with ONLY derived features (no price): {r2_derived:.6f}")





# Compare all models so far
results = pd.DataFrame({
    'Model': ['Naive Baseline', 'Linear (price only)', 
              'Random Forest', 'Linear (engineered)', 'Derived Only'],
    'R²': [0.8253, 0.8228, 0.6713, r2_new, r2_derived],
    'MAE': [14.28, 14.54, 20.02, mae_new, X_derived]
})

print(results.to_string(index=False))

# Compile results
results_df = pd.DataFrame({
    'Model': [
        'Naive Baseline',
        'Linear (Price Only)',
        'Random Forest',
        'Linear (Price + Engineered)',
        'Linear (Derived Only)'
    ],
    'R²': [
        0.8253,      # From Day 2
        0.8228,      # From Day 2
        0.6713,      # From Day 3
        0.8018,      # Day 4
        -0.5605      # Day 4
    ],
    'MAE': [
        14.28,       # From Day 2
        14.54,       # From Day 2
        20.02,       # From Day 3
        14.83,       # Day 4
        20.02        # Day 4
    ]
})

print(f"\n COMPLETE MODEL COMPARISON:")
print(results_df.to_string(index=False))

# Save to file
results_df.to_csv('results/model_comparison.csv', index=False)