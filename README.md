
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

## Results

### Linear Regression
- **Root Mean Squared Error**:    `27.54702361737279`
- **R-squared**:   `0.7302529920850762`
- **Mean Absolute Error**:    `21.367712812574894`

### Random Forest Regressor
- **Root Mean Squared Error**:    `20.846111018796346`
- **R-squared**:    `0.845525298011988`
- **Mean Absolute Error**:   `15.495057719298245`

