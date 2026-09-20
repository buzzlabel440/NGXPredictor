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
