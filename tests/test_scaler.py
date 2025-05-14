import pytest
import pandas as pd
import numpy as np
from sklite.preprocessing.scaler import *


@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({
        'age': [25, 30, 35, 40, 45],
        'income': [50000, 60000, 70000, 80000, 90000],
        'city': ['NY', 'LA', 'NY', 'SF', 'LA']
    })


def test_standar_scaler(sample_dataframe):
    scaler = StandardScaler(columns=['age', 'income'])
    scaler.fit(sample_dataframe)
    transformed = scaler.transform(sample_dataframe)

    # Check if the columns are transformed correctly
    age_transformed = (sample_dataframe['age'] - scaler.means['age']) / scaler.stds['age']
    income_transformed = (sample_dataframe['income'] - scaler.means['income']) / scaler.stds['income']

    assert np.allclose(transformed['age'], age_transformed)
    assert np.allclose(transformed['income'], income_transformed)
    assert 'age' in transformed.columns
    assert 'income' in transformed.columns
    assert transformed['age'].dtype == float or transformed['age'].dtype == 'float64'
    assert transformed['income'].dtype == float or transformed['income'].dtype == 'float64'
    assert 'city' not in scaler.means.keys()
    assert 'city' not in scaler.stds.keys()


def test_minmax_scaler(sample_dataframe):
    scaler = MinMaxScaler(columns=['age', 'income'])
    scaler.fit(sample_dataframe)
    transformed = scaler.transform(sample_dataframe)

    # Check if the columns are transformed correctly
    age_transformed = (sample_dataframe['age'] - np.min(sample_dataframe['age'])) / (np.max(sample_dataframe['age']) - np.min(sample_dataframe['age']))
    income_transformed = (sample_dataframe['income'] - np.min(sample_dataframe['income'])) / (np.max(sample_dataframe['income']) - np.min(sample_dataframe['income']))

    assert np.allclose(transformed['age'], age_transformed)
    assert np.allclose(transformed['income'], income_transformed)

    assert 'age' in transformed.columns
    assert 'income' in transformed.columns
    assert transformed['age'].dtype == float or transformed['age'].dtype == 'float64'
    assert transformed['income'].dtype == float or transformed['income'].dtype == 'float64'
    assert 'city' not in scaler.mins.keys()
    assert 'city' not in scaler.maxs.keys()