
# House Rent Prediction using Machine Learning

This project aims to predict house prices based on various features such as square footage, number of bedrooms, location(city),Point of Contact and Bathroom(s). Using regression techniques like Linear Regression and Random Forest, the model is trained on a real estate dataset to provide accurate price estimates.

## Features

- **Data Preprocessing**: Handling missing values, encoding categorical variables, and removing duplicates.
- **Exploratory Data Analysis (EDA)**: Visualizing data distributions and correlations.
- **Modeling**:
  - Linear Regression
  - Random Forest Regressor

## Dataset

The notebook uses a dataset named `House_Rent_Dataset.csv`. Ensure the dataset is placed in the appropriate directory (`../Data/`) for the notebook to access it.

## Requirements

The following libraries are used in the notebook:
```bash
pandas
numpy
scikit-learn
xgboost
seaborn
matplotlib
```

## Notebook Overview

### Data Preprocessing

- Removes duplicates and handles missing values.
- Encodes categorical variables using `LabelEncoder`.

### Exploratory Data Analysis

- Visualizes feature distributions using `seaborn` and `matplotlib`.
- Correlation matrix for feature importance.

### MODELS

#### Linear Regression

- Fits the model using `LinearRegression` from scikit-learn.
- Evaluates using Root Mean Squared Error (RMSE), R-squared, and Mean Absolute Error (MAE).

#### Random Forest Regressor

- Fits the model using `RandomForestRegressor`.
- Calculates feature importances.
- Evaluates using RMSE, R-squared, and MAE.
## Hyperparameter Tuning

### Random Forest Regressor - GridSearchCV

Used `GridSearchCV` to tune hyperparameters for the Random Forest Regressor. Key parameters to tune include:
- `n_estimators`: Number of trees in the forest.
- `max_depth`: Maximum depth of the trees.
- `min_samples_split`: Minimum samples required to split an internal node.
- `min_samples_leaf`: Minimum samples required at a leaf node.

GridSearchCV will help find the best parameter combination by performing cross-validation.

### Polynomial Regression - Degree Adjustment

In Linear Regression, you can improve the model by adding polynomial features. Experiment with different polynomial degrees to capture non-linear relationships and find the optimal degree for your dataset.

## Results

### Linear Regression
- **Root Mean Squared Error**:    `22.954355501747365`
- **R-squared**:   `0.8127001281522502`
- **Mean Absolute Error**:    `17.441181640625`

### Random Forest Regressor
- **Root Mean Squared Error**:    `17.64257036741746`
- **R-squared**:    `0.8923273302794676`
- **Mean Absolute Error**:   `13.051423306745969`

