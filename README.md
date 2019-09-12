# UncertaintyForecasting

This project is about making probabilistic calibration. The main
objectives are to provide:

- Probabilistic models
- Diagnostic tools
- Visualization tools
- Calibration tools

## pytest

```
python -m pytest -vv --cov-config=.coveragerc --cov-report html:cov_html --cov=. tests/
```

# 1 - Probabilistic models

- LSTM  + bayesian linear regression
- LSTM + dropout
- Linear classifier
- Linear classifer with temperatures

# 2 - Diagnostic tools

- Confidence interval calibration
- one-sided calibration
- cumulative distribution calibration
- SCE
- TACE (to be implemented)

# 3 - Visualization tools

- Calibration curves
- Regression with confidence interval

# 4 - Calibration tools

- UTC (to be implemented)
- Linear regressor (to be implemented)

# 5 - Examples

## 5.1 LSTM + Bayesian linear regression



## 5.2 Bayesian linear classification

## 5.3 LSTM + correlated dropout


