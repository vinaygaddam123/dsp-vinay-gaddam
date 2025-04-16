# house_prices/inference.py
import pandas as pd
import numpy as np
import joblib
from house_prices.preprocess import feature_engineering, conversion
from house_prices.preprocess import fill_values

# Columns
num_cols = ['TotalBathrooms', 'TotalPorchSF', 'HouseAge', 'RemodelAge',
            'GarageAge', 'LotFrontage', 'LotArea', 'GarageYrBlt', 'GarageArea',
            'GrLivArea']
cat_cols = ['LotShape', 'LotConfig', 'BldgType', 'HouseStyle', 'BsmtQual',
            'GarageType', 'GarageFinish', 'RoofStyle', 'Foundation',
            'SaleCondition', 'Utilities', 'LandSlope']


def make_predictions(input_data: pd.DataFrame) -> np.ndarray:
    """
    Make predictions on new input data using a saved model
    and preprocessing objects.

    Parameters:
        input_data (pd.DataFrame): Input dataframe for prediction.

    Returns:
        np.ndarray: Array of predicted values.
    """
    model = joblib.load("models/model.joblib")
    scaler = joblib.load("models/scaler.joblib")
    encoder = joblib.load("models/encoder.joblib")

    data = feature_engineering(input_data)
    data = conversion(data)
    data = fill_values(data, num_cols, cat_cols)

    data_num = scaler.transform(data[num_cols])
    data_cat = encoder.transform(data[cat_cols]).toarray()

    encoded_cols = encoder.get_feature_names_out(cat_cols)
    data_final = pd.DataFrame(data_num,
                              columns=num_cols).reset_index(drop=True)
    data_cat_df = pd.DataFrame(data_cat,
                               columns=encoded_cols).reset_index(drop=True)
    data_final = pd.concat([data_final, data_cat_df], axis=1)

    predictions = model.predict(data_final)

    predicted_prices = pd.DataFrame(predictions, columns=['PredictedPrice'])

    return predicted_prices
