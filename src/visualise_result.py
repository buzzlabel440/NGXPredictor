import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

# Load your data and models from Day 4
df = pd.read_csv('data/raw/stock_data.csv', index_col='Date', parse_dates=True)
results_df = pd.read_csv('results/model_comparison.csv')

# Assume you have: y_test, y_pred_linear, y_pred_naive, y_test_actual
# (Reload from your saved models or recalculate)

# Create figure with 4 subplots
fig = plt.figure(figsize=(16, 12))

# ===== PLOT 1: Model Comparison (R² Scores) =====
ax1 = plt.subplot(2, 2, 1)
models = results_df['Model'].values
r2_scores = results_df['R²'].values
colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6']

bars = ax1.barh(models, r2_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax1.set_xlabel('R² Score', fontsize=12, fontweight='bold')
ax1.set_title('Model Performance Comparison\n(Higher is Better)', fontsize=13, fontweight='bold')
ax1.set_xlim(0, 1)
ax1.grid(axis='x', alpha=0.3, linestyle='--')

# Add value labels on bars
for i, (bar, score) in enumerate(zip(bars, r2_scores)):
    if score > 0:
        ax1.text(score + 0.02, i, f'{score:.4f}', va='center', fontweight='bold')
    else:
        ax1.text(score - 0.05, i, f'{score:.4f}', va='center', ha='right', fontweight='bold', color='red')

# Highlight naive baseline
ax1.axvline(0.8253, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Naive Baseline')
ax1.legend()

# ===== PLOT 2: MAE Comparison =====
ax2 = plt.subplot(2, 2, 2)
mae_scores = results_df['MAE'].values
bars2 = ax2.barh(models, mae_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax2.set_xlabel('Mean Absolute Error ($)', fontsize=12, fontweight='bold')
ax2.set_title('Prediction Error Comparison\n(Lower is Better)', fontsize=13, fontweight='bold')
ax2.grid(axis='x', alpha=0.3, linestyle='--')

# Add value labels
for bar, mae in zip(bars2, mae_scores):
    ax2.text(mae + 0.5, bar.get_y() + bar.get_height()/2, f'${mae:.2f}', 
             va='center', fontweight='bold')

# ===== PLOT 3: Autocorrelation Visualization =====
ax3 = plt.subplot(2, 2, 3)
# Scatter plot: close(t) vs close(t+1)
close_prices = df['Close'].dropna()
close_today = close_prices[:-1].values
close_tomorrow = close_prices[1:].values

ax3.scatter(close_today, close_tomorrow, alpha=0.5, s=30, color='#3498db', edgecolor='black', linewidth=0.5)
# Add perfect prediction line
min_price = min(close_today.min(), close_tomorrow.min())
max_price = max(close_today.max(), close_tomorrow.max())
ax3.plot([min_price, max_price], [min_price, max_price], 'r--', linewidth=2, label='Perfect Prediction')
ax3.set_xlabel('Close Price Today ($)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Close Price Tomorrow ($)', fontsize=12, fontweight='bold')
ax3.set_title('Autocorrelation Evidence\n(r = 0.9806)', fontsize=13, fontweight='bold')
ax3.legend()
ax3.grid(alpha=0.3)

# Add correlation text
corr = np.corrcoef(close_today, close_tomorrow)[0, 1]
ax3.text(0.05, 0.95, f'Correlation: {corr:.6f}', transform=ax3.transAxes, 
         fontsize=11, fontweight='bold', verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

# ===== PLOT 4: Actual vs Predicted (Best Model) =====
ax4 = plt.subplot(2, 2, 4)
# You'll need to reload y_test and y_pred from your Day 2 model
# For now, showing conceptual code:
# ax4.plot(y_test.values[:50], 'o-', label='Actual', linewidth=2, markersize=6)
# ax4.plot(y_pred[:50], 's--', label='Predicted (Linear)', linewidth=2, markersize=6, alpha=0.7)

# If not saved, recreate:
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df_sorted = df.sort_values('Date')
X = df_sorted[['Close', 'High', 'Low', 'Volume']].shift(1).dropna()
y = df_sorted['Close'].shift(-1).dropna()
min_len = min(len(X), len(y))
X, y = X[:min_len], y[:min_len]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# Plot last 50 days of test set
ax4.plot(y_test.values[-50:], 'o-', label='Actual', linewidth=2.5, markersize=6, color='#e74c3c')
ax4.plot(y_pred[-50:], 's--', label='Predicted (Linear)', linewidth=2.5, markersize=6, 
         alpha=0.7, color='#3498db')
ax4.set_xlabel('Day (Last 50 Days of Test Set)', fontsize=12, fontweight='bold')
ax4.set_ylabel('Close Price ($)', fontsize=12, fontweight='bold')
ax4.set_title('Actual vs Predicted Price\n(Linear Regression Model)', fontsize=13, fontweight='bold')
ax4.legend(fontsize=11)
ax4.grid(alpha=0.3)

# Add R² and MAE to plot
r2_test = r2_score(y_test, y_pred)
mae_test = mean_absolute_error(y_test, y_pred)
ax4.text(0.05, 0.95, f'R²: {r2_test:.4f}\nMAE: ${mae_test:.2f}', 
         transform=ax4.transAxes, fontsize=11, fontweight='bold', verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

# Overall layout
fig.suptitle('NSE Stock Price Predictor: Complete Analysis\nInfosys (INFY.NS)', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.99])

# Save
plt.savefig('results/day5_visualizations.png', dpi=300, bbox_inches='tight')
print("Visualization saved to results/day5_visualizations.png")
plt.show()