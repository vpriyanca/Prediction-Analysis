# University Enrollment Prediction Project

## Overview
This project focuses on predicting student enrollment at a university using historical data from various academic, demographic, and institutional factors. The goal is to build predictive models that support strategic planning and optimize resource allocation in higher education. The project employs statistical modeling, machine learning algorithms, and time-series techniques to forecast enrollment at campus and class levels.

## Features
- Data ingestion and preprocessing of multi-year enrollment records.
- Feature selection from demographics, residency, course load, and funding categories.
- Time-series modeling for historical enrollment patterns.
- Machine learning modeling using:
  - Random Forest
	- Gradient Boosting
	- ARIMA
	- LSTM
- Ensemble modeling to improve prediction accuracy.
- Evaluation using metrics such as MAD (Mean Absolute Deviation) and MSE (Mean Squared Error).
- Export of comparative model performance results.
- Visual insights via trend lines and error charts.

## Installation
To run this project, ensure Python is installed along with necessary libraries. You can create a virtual environment and install dependencies as shown below:
```
# Clone the repository
git clone https://github.com/yourusername/University-Enrollment-Prediction.git
cd University-Enrollment-Prediction

# (Optional) Create a virtual environment
python -m venv enroll-env
source enroll-env/bin/activate  # On Windows: enroll-env\Scripts\activate

# Install required packages
pip install -r requirements.txt
```
## Usage
This project is structured to predict student enrollment using both time-series and machine learning approaches.

### Data Preparation
The dataset includes columns such as:
- Fiscal Year, Term, Campus
- Student demographics (age, gender, residency, ethnicity)
- Academic indicators (credits taken, full-time/part-time, degree-seeking)
- Financial indicators (state funding, tuition fee category)
- Enrollment labels (campus FTE, KSU FTE, fully online, first-generation)

The primary source file is expected to be placed under the data/ directory.

### Running the Code
You can run the analysis in the notebook Enrollment_Prediction_copy.ipynb. Key steps include:
1. Preprocessing and filtering of data.
2. Creating separate datasets for training and testing (e.g., holding 2023 as the test set).
3. Running machine learning and time series models.
4. Evaluating predictions with MAD and MSE.
5. Visualizing and comparing results across campuses and class levels.

### Models
**Machine Learning Models**
- Random Forest
- Gradient Boosting
- Ensemble Averaging

**Time-Series Models**
- ARIMA
- LSTM

Each model is tuned and validated using historical data, with predictions generated for the most recent year (2023).

### Visualization
The following charts are used for analysis:
- **Trend Line Charts**: Compare actual vs. predicted enrollment by year and campus.
- **Error Distribution Plots**: Visualize MAD and MSE across models.
- **Feature Importance**: View top predictors contributing to model accuracy.

### Custom Functions
The notebook includes utility functions to:
- Split datasets by campus (KC vs. non-KC)
- Calculate evaluation metrics (MAD, MSE)
- Create ensemble predictions
- Export summarized results

## Contributing
If you’d like to improve this project:
1. Fork the repository.
2. Create a new branch (git checkout -b feature-branch).
3. Commit your changes.
4. Push to your fork.
5. Submit a pull request.

For major contributions, please open an issue first to discuss your ideas.

## Acknowledgment
This project was developed at Kent State University as part of a strategic data science initiative. Thanks to the data management and analytics team, and tools including:
- Python
- Pandas, NumPy
- Scikit-learn
- TensorFlow / Keras
- Statsmodels
- Matplotlib, Seaborn
- JMP (for validation and export)
