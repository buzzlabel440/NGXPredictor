Creating a machine learning model that predict NSE price.

=============Two models are used==========
LinearRegression and Random Forest

so far, LinearRegression is showing auto correlation in its prediction


## Model Performance

The naive baseline (predict tomorrow = today) achieves R² = 0.8253.
Our linear regression achieves R² = 0.8228.

Conclusion: Stock prices exhibit 0.9806 autocorrelation. 
There is insufficient signal for a predictive model to improve on 
the baseline. The problem is inherently noisy.


## Day 3: Random Forest Analysis

 ### Results
 R2 Score: 0.671262
 MEA_RF: 20.02
 Close: 0.36
 Low: 0.35
 High: 0.28
 Volume: 0.01


 ## Day 3: Random Forest Analysis

### Finding
Random Forest R² = 0.6713, MAE = $20.02
Conclusion: Massive overfitting. Model fails to generalize.

### Why
- Autocorrelation (0.9806) leaves <2% signal
- RF fit noise, not signal
- Test set exposed: 15% R² gap between train/test
- Feature importance balanced (noise absorption pattern)

### Lesson
Ensemble methods don't guarantee improvement. When signal is weak, 
complexity amplifies overfitting. The naive model (R² = 0.8253) 
remains champion.

### Next Step
To improve beyond 0.83 R², we need information outside price data.




# NSE Stock Price Predictor 📈

A rigorous 7-day machine learning project predicting Infosys (INFY.NS) closing prices.
Built to understand market efficiency, not maximize accuracy.

## 🎯 Project Goal

Can we predict next-day NSE stock prices using historical price data?

Short answer: Technically yes, but practically no.

Long answer: Read below.

---

## 📊 Key Findings

### The Autocorrelation Ceiling

Infosys closing prices exhibit 0.9806 autocorrelation between consecutive days.

This means: 98% of tomorrow's price is already encoded in today's price.

### Model Performance Rankings

| Rank | Model | R² Score | MAE | Notes |
|------|-------|----------|-----|-------|
| 🥇 | Naive Baseline | 0.8253 | $14.28 | Winner: tomorrow = today |
| 🥈 | Linear (Price Only) | 0.8228 | $14.54 | Slight underperformance |
| 🥉 | Linear (Price + Features) | 0.8018 | $14.83 | Added noise, hurt performance |
| ❌ | Linear (Derived Only) | -0.5605 | ~$X | Catastrophic failure |
| ❌ | Random Forest | 0.6713 | $20.02 | Severe overfitting |

Conclusion: Adding complexity reduces performance. The naive baseline is optimal.

---

## 🧠 Why This Happened

### The Efficient Market Hypothesis (EMH)

Stock markets efficiently incorporate available information into prices. Therefore:

1. Yesterday's price already contains all past information
2. Technical indicators are lagging—they describe, not predict
3. Feature engineering on price-derived data adds noise, not signal
4. Only exogenous events (earnings, macro data, sentiment) can predict

### Mathematical Foundation

$$R^2_{\text{max}} = r^2 = (0.9806)^2 ≈ 0.9616$$

The theoretical ceiling for any model using only price data is ~96.16% R².
The naive baseline already captures 82.53%—a massive fraction of what's possible.

---

## 🔬 Methodology

### Data Source
- Stock: Infosys Limited (INFY.NS)
- Period: 1 year of daily data (252 trading days)
- Source: Yahoo Finance API

### Feature Engineering Tested
1. Original features: Close, High, Low, Volume
2. Engineered features: 
   - 20-day rolling volatility
   - Intraday range (high - low) %
   - Volume ratio (today vs 20-day avg)
   - Daily returns (momentum)
   - Close-Open range %
   - RSI-14 (Relative Strength Index)
3. Ensemble methods: Random Forest (100 trees)

### Train/Test Split
- Training: 80% (Days 1-200)
- Testing: 20% (Days 201-252)
- Order preserved: No shuffling (time-series data)

### Models Compared
1. Naive Baseline: Predict tomorrow = today
2. Linear Regression: 4 features (price-based)
3. Linear Regression: 10 features (price + engineered)
4. Linear Regression: 6 features (engineered only, no price)
5. Random Forest: 100 trees, max_depth=10

---

## 📈 Visualizations

See results/day5_visualizations.png for:
- Model performance comparison (R² and MAE)
- Autocorrelation scatter plot (proof of 0.9806 r)
- Actual vs predicted prices (best model)
- Feature importance analysis

---

## 💡 Lessons Learned

### 1. Understand Data Structure Before Modeling
Autocorrelation is a structural property, not noise. No amount of feature engineering overcomes it.

### 2. Simplicity Beats Complexity When Signal is Weak
- Linear model (0.8228) > Random Forest (0.6713)
- Naive baseline (0.8253) > Engineered features (0.8018)
- Occam's Razor holds in ML.

### 3. Test Sets Prevent False Confidence
Random Forest looked promising on training data but catastrophically failed on test data. 
This gap revealed overfitting—a lesson that wouldn't have emerged without proper validation.

### 4. Market Efficiency is Real
EMH isn't just theory—it's empirically provable. You can't beat the market with price data alone.

### 5. Communication > Accuracy
An honest "we can't predict this" is more valuable than an overfit model claiming 95% accuracy.

---

## 🚀 How to Run

### Prerequisites
```bash
pip install pandas numpy scikit-learn yfinance matplotlib