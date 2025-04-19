# Credit Card Default Payment Prediction

## Project Overview

This project aims to predict whether a credit card customer will default on their next payment using machine learning classification techniques. Two models are explored:

- **Gaussian Naive Bayes**
- **XGBoost Classifier**

The workflow includes data exploration, preprocessing, model training, hyperparameter tuning, and evaluation.

## Dataset Description

The dataset file is named `creditCardFraud_28011964_120214.csv`. It contains 30,000 records and 25 columns, including the target variable.

Key columns include:

- **limit\_balance**: Amount of given credit (NT dollar)
- **sex**: Gender (1 = male, 2 = female)
- **education**: Education level (1 = graduate school; 2 = university; 3 = high school; 4 = others)
- **marriage**: Marital status (1 = married; 2 = single; 3 = others)
- **age**: Age in years
- **pay\_status1** ... **pay\_status6**: Repayment status in September 2005 to April 2005 (scale: –2 to 8)
- **bill\_amt1** ... **bill\_amt6**: Amount of bill statement in September 2005 to April 2005 (NT dollar)
- **pay\_amt1** ... **pay\_amt6**: Amount of previous payment in September 2005 to April 2005 (NT dollar)
- **default payment next month**: Target variable (1 = default, 0 = no default)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```
2. Create a Python environment (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Ensure the dataset CSV (`creditCardFraud_280119.csv`) is located in the project root.
2. Launch Jupyter Notebook:
   ```bash
   jupyter notebook creditcarrd_fraud_detection.ipynb
   ```
3. Run the notebook cells sequentially to reproduce the analysis and results.

## Methodology

1. **Data Exploration**

   - Loaded the data and inspected basic statistics (head, shape, info, missing values).
   - Visualized relationships with a pairplot and correlation heatmap.

2. **Data Preprocessing**

   - Defined feature matrix `X` by dropping the target column.
   - Defined label vector `y` as the target column (`default payment next month`).
   - Split the data into training and testing sets (33% test size, random state = 42).
   - Standardized the features using `StandardScaler`.

3. **Model Building & Evaluation**

   - **Gaussian Naive Bayes (GNB)**

     - Trained a baseline GNB model and evaluated accuracy.
     - Tuned the `var_smoothing` parameter using `GridSearchCV`.
     - Retrained GNB with the best smoothing value and compared performance.

   - **XGBoost Classifier**

     - Defined a hyperparameter grid for `n_estimators`, `max_depth`, and `random_state`.
     - Performed grid search with 5-fold cross-validation.
     - Evaluated the tuned XGBoost model on the test set.

## Evaluation Metrics

- **Accuracy Score**: Percentage of correctly predicted default/non-default cases.

*(Detailed classification reports and confusion matrices can be found within the Jupyter notebook.)*

## Results

- **Baseline Gaussian Naive Bayes Accuracy**: 77.02%
- **Tuned Gaussian Naive Bayes Accuracy**: 77.02%
- **Tuned XGBoost Accuracy**: 81.12%

The XGBoost Classifier outperformed the Gaussian Naive Bayes model with a higher accuracy after hyperparameter tuning.

## Future Work

- Experiment with additional classifiers (e.g., Random Forest, Logistic Regression).
- Incorporate techniques for imbalanced data (e.g., SMOTE, class weighting).
- Analyze feature importance and remove redundant features.
- Deploy the best model with a Flask or Streamlit web application.

## Dependencies

Required Python packages:

```
pandas
numpy
seaborn
matplotlib
scikit-learn
xgboost
jupyter
```

You can install them via:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn xgboost jupyter
```

## License

This project is licensed under the MIT License. You are free to use, modify, and distribute this project with proper attribution. See the LICENSE file for full details.

