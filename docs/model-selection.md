\# Model Selection



\## Purpose



This document records how the first production model was selected for the Retail Demand Intelligence pipeline.



The project evaluates two forecasting strategies:



\- Poisson demand model

\- Two-stage demand model



\## Training configuration



\- Model version: `run-20261011T163019Z`

\- Forecast horizon: 28 days

\- Validation period: 28 days

\- Test period: 28 days

\- Demand threshold: `0.50`

\- Training rows: 5,576,621

\- Validation rows: 85,372

\- Test rows: 85,372

\- Total feature rows: 5,918,109

\- Model features: 30



The training split was used to fit the models.



The validation split was used for early stopping.



The final test split remained untouched until the final evaluation.



\## Validation results



| Model | MAE | RMSE | WMAPE | Bias |

|---|---:|---:|---:|---:|

| Poisson | 1.115 | 2.166 | 74.4% | +1.94% |

| Two-stage | 1.036 | 2.223 | 69.2% | -15.62% |



The two-stage model produced lower MAE and WMAPE on validation data, but it showed meaningful underforecasting bias.



\## Final holdout evaluation



The final test period was:



\- Start: `2016-04-25`

\- End: `2016-05-22`

\- Rows: 85,372

\- Available rows: 85,372



| Model | MAE | RMSE | WMAPE | Bias |

|---|---:|---:|---:|---:|

| Poisson | 1.155 | 2.267 | 73.6% | -4.45% |

| Two-stage | 1.099 | 2.333 | 70.0% | -21.25% |



\## Production decision



The Poisson model is selected as the first production candidate.



The decision is based on:



\- Lower RMSE on the untouched test period

\- Much safer aggregate bias

\- More balanced total-demand forecasts

\- More reliable behavior for demand-planning decisions



The two-stage model has slightly better MAE and WMAPE, but its 21.25% negative bias indicates substantial underforecasting. In a retail planning context, this could contribute to stockouts and insufficient replenishment.



The two-stage model remains available as a challenger model for future experimentation.



\## Model artifacts



The versioned model artifacts are stored locally under:



```text

artifacts/models/run-20261011T163019Z/