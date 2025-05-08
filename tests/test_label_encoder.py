import pandas as pd
import pytest
from sklite.preprocessing.label_encoding import LabelEncoder


@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({
        'gender': ['M', 'F', 'F', 'M', 'F'],
        'city': ['NY', 'LA', 'NY', 'SF', 'LA']
    })

def test_fit_creates_label_map(sample_dataframe):
    encoder = LabelEncoder(columns=['gender'])
    encoder.fit(sample_dataframe)

    assert encoder.label_maps['gender'] == {'M': 0, 'F': 1} or encoder.label_maps['gender'] == {'F': 0, 'M': 1}

def test_transform_converts_values(sample_dataframe):
    encoder = LabelEncoder(columns=['city'])
    encoder.fit(sample_dataframe)
    result = encoder.transform(sample_dataframe)

    assert 'city' in result.columns
    assert result['city'].dtype == int or result['city'].dtype == 'int64'
    assert set(result['city'].unique()).issubset(set(range(len(sample_dataframe['city'].unique()))))

def test_fit_transform_equivalence(sample_dataframe):
    encoder = LabelEncoder(columns=['gender'])
    fit_then_transform = encoder.fit(sample_dataframe).transform(sample_dataframe)

    encoder2 = LabelEncoder(columns=['gender'])
    fit_transform = encoder2.fit_transform(sample_dataframe)

    pd.testing.assert_frame_equal(fit_then_transform, fit_transform)

def test_inverse_transform_restores_original(sample_dataframe):
    encoder = LabelEncoder(columns=['gender'])
    transformed = encoder.fit_transform(sample_dataframe)
    recovered = encoder.inverse_transform(transformed)

    assert recovered['gender'].tolist() == sample_dataframe['gender'].tolist()
