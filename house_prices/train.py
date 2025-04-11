# house_prices/train.py
import os
import pandas as pd
import numpy as np
import joblib
from typing import Dict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from house_prices.preprocess import feature_engineering, conversion
from house_prices.preprocess import fill_values

# Columns
num_cols = ['TotalBathrooms', 'TotalPorchSF', 'HouseAge', 'RemodelAge',
            'GarageAge', 'LotFrontage', 'LotArea', 'GarageYrBlt',
            'GarageArea', 'GrLivArea']
cat_cols = ['LotShape', 'LotConfig', 'BldgType', 'HouseStyle',
            'BsmtQual', 'GarageType', 'GarageFinish', 'RoofStyle',
            'Foundation', 'SaleCondition', 'Utilities', 'LandSlope']


def build_model(data: pd.DataFrame) -> Dict[str, float]:
    """
    Train a linear regression model on the input dataset and
    save the model and preprocessors.

    Parameters:
        data (pd.DataFrame): Full training dataframe with
        target column `SalePrice`.

    Returns:
        Dict[str, float]: Dictionary containing model
        performance metrics (RMSE).
    """
    X = data.drop(columns=['SalePrice'])
    y = data['SalePrice']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42)

    # Preprocessing
    X_train = feature_engineering(X_train)
    X_train = conversion(X_train)
    X_train = fill_values(X_train, num_cols, cat_cols)

    scaler = StandardScaler()
    scaler.fit(X_train[num_cols])
    X_train_num = scaler.transform(X_train[num_cols])

    encoder = OneHotEncoder(handle_unknown='ignore')
    encoder.fit(X_train[cat_cols])
    X_train_cat = encoder.transform(X_train[cat_cols]).toarray()

    encoded_cols = encoder.get_feature_names_out(cat_cols)
    X_train_final = pd.DataFrame(X_train_num,
                                 columns=num_cols).reset_index(drop=True)
    X_train_cat_df = pd.DataFrame(X_train_cat,
                                  columns=encoded_cols).reset_index(drop=True)
    X_train_final = pd.concat([X_train_final, X_train_cat_df], axis=1)

    model = LinearRegression()
    model.fit(X_train_final, y_train)

    # Save everything
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.joblib")
    joblib.dump(scaler, "models/scaler.joblib")
    joblib.dump(encoder, "models/encoder.joblib")

    # Evaluation
    X_test = feature_engineering(X_test)
    X_test = conversion(X_test)
    X_test = fill_values(X_test, num_cols, cat_cols)

    X_test_num = scaler.transform(X_test[num_cols])
    X_test_cat = encoder.transform(X_test[cat_cols]).toarray()

    X_test_final = pd.DataFrame(X_test_num,
                                columns=num_cols).reset_index(drop=True)
    X_test_cat_df = pd.DataFrame(X_test_cat,
                                 columns=encoded_cols).reset_index(drop=True)
    X_test_final = pd.concat([X_test_final, X_test_cat_df], axis=1)

    y_pred = model.predict(X_test_final)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    return {'rmse': round(rmse, 2)}
