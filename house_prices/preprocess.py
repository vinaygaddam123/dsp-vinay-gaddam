# house_prices/preprocess.py

import pandas as pd
from typing import List


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add new engineered features to the input dataframe.

    Parameters:
        df (pd.DataFrame): Raw input dataframe.

    Returns:
        pd.DataFrame: Dataframe with new engineered features.
    """
    df = df.copy()

    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df['TotalBathrooms'] = (
        df['FullBath'] + 0.5 * df['HalfBath'] +
        df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath']
    )
    df['TotalPorchSF'] = (
        df['OpenPorchSF'] + df['EnclosedPorch'] +
        df['3SsnPorch'] + df['ScreenPorch']
    )
    df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    df['RemodelAge'] = df['YrSold'] - df['YearRemodAdd']
    df['GarageAge'] = df['YrSold'] - df['GarageYrBlt']
    df['HasGarage'] = df['GarageType'].notnull().astype(int)
    df['IsRemodeled'] = (df['YearBuilt'] != df['YearRemodAdd']).astype(int)

    qual_mapping = {'Ex': 5, 'Gd': 4, 'TA': 3,
                    'Fa': 2, 'Po': 1, None: 0}
    for col in ['ExterQual', 'KitchenQual', 'FireplaceQu',
                'BsmtQual', 'HeatingQC']:
        df[col + '_Num'] = df[col].map(qual_mapping)

    return df


def conversion(data: pd.DataFrame) -> pd.DataFrame:
    """
    Convert object-type columns to category.

    Parameters:
        data (pd.DataFrame): Input dataframe.

    Returns:
        pd.DataFrame: Dataframe with category dtype columns.
    """
    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = data[col].astype('category')
    return data


def fill_values(data: pd.DataFrame,
                num_cols: List[str],
                cat_cols: List[str]) -> pd.DataFrame:
    """
    Fill missing values for numerical and categorical columns.

    Parameters:
        data (pd.DataFrame): Input dataframe.
        num_cols (List[str]): List of numerical columns.
        cat_cols (List[str]): List of categorical columns.

    Returns:
        pd.DataFrame: Cleaned dataframe with no missing values.
    """
    data.fillna({col: data[col].mean() for col in num_cols}, inplace=True)
    data.fillna({col: data[col].mode()[0] for col in cat_cols}, inplace=True)
    return data
