# Day 4: Feature Engineering Experiment

## Hypothesis
Engineered features (volatility, RSI, momentum) can improve predictions
beyond simple price-based models.

## Results
- Linear (price only): R² = 0.8228
- Linear (price + engineered): R² = 0.8018 (WORSE)
- Linear (derived only): R² = -0.5605 (CATASTROPHIC)

## Conclusion
Engineered features add noise, not signal. The problem is inherently
autocorrelation-dominated. To improve beyond R² ≈ 0.825, we need
information sources outside price data (earnings, macro data, sentiment).

## Implication for Days 5-7
Simple price-based models are near-optimal for this problem. Pursuing
marginal gains through feature engineering is futile. Instead, focus on
honest documentation and visualization of the journey.
EOF