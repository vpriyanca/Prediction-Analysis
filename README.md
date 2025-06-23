# University Enrollment Prediction Project

## Overview
This project focuses on forecasting student enrollment at a university using time-series forecasting techniques and historical demographic data. The primary goal is to model and predict enrollment counts based on fiscal year data, age, residency, international status, and ethnicity. The project uses ARIMA and LSTM models, compares them with rolling averages, and outputs detailed demographic breakdowns.

## Features
- Cleans and preprocesses multi-year enrollment data from Excel.
- Converts term data to fiscal years and extracts demographic indicators.
- Aggregates and visualizes year-wise trends.
- Implements time-series forecasting models:
  - ARIMA (classical statistical modeling)
  - LSTM (deep learning-based sequence modeling)
- Calculates rolling averages to benchmark forecasts.
- Provides detailed prediction breakdowns by:
  - Age
  - International student ratio
  - In-state residency rate
  - Ethnicity code distribution
- Ensemble forecasting using ARIMA and LSTM predictions.

## Installation
To run this project, ensure you have the following installed in your Python environment:
```
pip install pandas numpy matplotlib scikit-learn statsmodels openpyxl tensorflow
```
If using Google Colab, just upload the notebook and Excel file, and all dependencies can be installed inline

## Usage
### Data Preparation
- Load the file: enrollment_data.xlsx
- Preprocesses columns such as:
  - FallTerm → FiscalYear
  - EthnicityCode (converted into dummy columns)
- Aggregates year-wise statistics:
  - Total student count
  - Average age
  - Proportion of international/in-state students
  - Ethnicity distribution (codes A, B, F, H, M, N, P, W, X)

### Running the Models
**ARIMA Forecasting**
- Model trained on student count time series (CountofStudents).
- Forecasts enrollment for selected years.
- Uses statsmodels ARIMA (1,1,1) configuration.

**LSTM Forecasting**
- Sequence model trained on 2-year windows.
- Predicts next year’s student count using scaled data.
- Uses tensorflow.keras LSTM layers.

**Ensemble Prediction**
- Final forecast for 2024 is averaged from ARIMA and LSTM outputs.
- Forecast includes all demographic proportions based on 2023 data.

### Evaluation Metrics

| Year | Model | Actual Student Count | Forecasted Count | Difference |
|------|--------|-----------------------|------------------|------------|
| 2016 | ARIMA  | 40,782                | 41,015.44        | +233.44    |
| 2016 | LSTM   | 40,782                | 41,002.75        | +220.75    |
| 2019 | ARIMA  | 37,411                | 38,325.97        | +914.97    |
| 2019 | LSTM   | 37,411                | 38,315.84        | +904.84    |
| 2022 | ARIMA  | 33,209                | 34,949.74        | +1,740.74  |
| 2022 | LSTM   | 33,209                | 34,752.47        | +1,543.47  |
| 2024 | ARIMA  | _N/A_                 | 33,463.80        | _N/A_      |
| 2024 | LSTM   | _N/A_                 | 33,431.87        | _N/A_      |
| 2024 | Ensemble | _N/A_               | 33,447.83        | _N/A_      |

### Visualization
The notebook includes the following visual outputs:
- **Rolling Average Plot**: 3-year smoothed line over student counts.
- **Forecast vs Actual**: Printed comparisons by year.
- **Predictions for 2024**: Includes demographic breakdowns alongside forecasted enrollment.
  
### Custom Functions
- predict_and_validate_arima(): ARIMA forecast with comparison to actuals.
- predict_and_validate_lstm(): Builds and trains LSTM model per prediction year.
- forecast_detailed_2024(): Combines model output and 2023 demographic ratios for 2024 prediction.
- print_detailed_predictions(): Human-readable summary for each year’s student profile.

### Key Learnings
- **Historical patterns are predictive**: Multi-year student count trends—when smoothed using rolling averages—show stable signals that can be used as baselines to validate forecasts.
- **Demographics add interpretability**: Including features like age, residency, and ethnicity enables more than just enrollment prediction; it supports strategic planning around diversity, outreach, and funding.
- **ARIMA vs. LSTM**:
  - ARIMA performed well for short-term trends but is sensitive to sudden shifts (e.g., post-pandemic enrollment dips).
  - LSTM better captured nonlinear patterns and recent changes but required more training data and tuning.
- **Ensemble modeling improves reliability**: Averaging predictions from ARIMA and LSTM gave a balanced forecast that leverages the stability of statistical models and the adaptability of deep learning.
- **Data quality matters**: Missing values in term-year or demographic codes caused issues during preprocessing. Cleaning and transforming these early on was critical to avoid bias and misalignment in the results.
- **Interpretability vs. accuracy trade-off**: Classical models like ARIMA offer high interpretability and are easy to explain to stakeholders. LSTM, while potentially more accurate, is less transparent and requires careful architecture design.
- **Forecasting beyond 2023** should always be paired with domain insights—especially when real-world events (e.g., policy shifts, tuition changes) can cause disruptions not reflected in historical data.

## Contributing
If you’d like to improve this project:
1. Fork the repository.
2. Create a new branch (git checkout -b feature-branch).
3. Commit your changes.
4. Push to your fork.
5. Submit a pull request.

For major contributions, please open an issue first to discuss your ideas.

## Contact
- **Project Owner**: Priyanka Vyas
- **Email**: vpriyanka.sv@gmail.com

## Acknowledgment
This project was developed at Kent State University as part of a strategic data science initiative. Thanks to the data management and analytics team, and tools including:
**Language & IDE:**  
![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
**Libraries & Frameworks:**  
![Pandas](https://img.shields.io/badge/Pandas-1.x-150458?logo=pandas)  
![NumPy](https://img.shields.io/badge/NumPy-1.x-013243?logo=numpy)  
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-F7931E?logo=scikit-learn)  
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow)  
![Keras](https://img.shields.io/badge/Keras-2.x-D00000?logo=keras)  
![Statsmodels](https://img.shields.io/badge/Statsmodels-Enabled-green)  
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.x-3776AB?logo=python)  
![Seaborn](https://img.shields.io/badge/Seaborn-0.11+-579ACA?logo=python)
**Other Tools:**  
![JMP](https://img.shields.io/badge/JMP-Validation%20%26%20Export-blue?logo=sas)
**Development Tool:**  
![Jupyter Notebook](https://img.shields.io/badge/Jupyter%20Notebook-Enabled-F37626?logo=jupyter)
**Hardware:**  
![MacBook Pro](https://img.shields.io/badge/MacBook%20Pro-Apple%20M1%20Pro-lightgrey?logo=apple&logoColor=white)

