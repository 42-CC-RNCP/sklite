import pytest
import numpy as np
from sklite.metrics import *


@pytest.fixture
def sample_data():
    y_true = np.array([3, -0.5, 2, 7])
    y_pred = np.array([2.5, 0.0, 2, 8])
    return y_true, y_pred


def test_mean_squared_error(sample_data):
    y_true, y_pred = sample_data
    mse_metric = MeanSquaredError()
    mse_value = mse_metric(y_true, y_pred)

    expected_mse = np.mean((y_true - y_pred) ** 2)
    assert np.isclose(mse_value, expected_mse), f"Expected {expected_mse}, but got {mse_value}"
    assert isinstance(mse_value, float), "MSE value should be a float"
    assert mse_metric._name == 'Mean Squared Error', "Metric name should be 'Mean Squared Error'"
    assert mse_metric._description == 'Mean Squared Error (MSE) metric.', "Metric description should be 'Mean Squared Error (MSE) metric.'"


def test_r2_score(sample_data):
    y_true, y_pred = sample_data
    r2_metric = R2Score()
    r2_value = r2_metric(y_true, y_pred)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    expected_r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

    assert np.isclose(r2_value, expected_r2), f"Expected {expected_r2}, but got {r2_value}"
    assert isinstance(r2_value, float), "R^2 value should be a float"
    assert r2_metric._name == 'R^2 Score', "Metric name should be 'R^2 Score'"
    assert r2_metric._description == 'R^2 Score metric.', "Metric description should be 'R^2 Score metric.'"


def test_mean_absolute_error(sample_data):
    y_true, y_pred = sample_data
    mae_metric = MeanAbsoluteError()
    mae_value = mae_metric(y_true, y_pred)

    expected_mae = np.mean(np.abs(y_true - y_pred))
    assert np.isclose(mae_value, expected_mae), f"Expected {expected_mae}, but got {mae_value}"
    assert isinstance(mae_value, float), "MAE value should be a float"
    assert mae_metric._name == 'Mean Absolute Error', "Metric name should be 'Mean Absolute Error'"
    assert mae_metric._description == 'Mean Absolute Error (MAE) metric.', "Metric description should be 'Mean Absolute Error (MAE) metric.'"


def test_root_mean_squared_error(sample_data):
    y_true, y_pred = sample_data
    rmse_metric = RootMeanSquaredError()
    rmse_value = rmse_metric(y_true, y_pred)

    expected_rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    assert np.isclose(rmse_value, expected_rmse), f"Expected {expected_rmse}, but got {rmse_value}"
    assert isinstance(rmse_value, float), "RMSE value should be a float"
    assert rmse_metric._name == 'Root Mean Squared Error', "Metric name should be 'Root Mean Squared Error'"
    assert rmse_metric._description == 'Root Mean Squared Error (RMSE) metric.', "Metric description should be 'Root Mean Squared Error (RMSE) metric.'"


def test_mean_absolute_percentage_error(sample_data):
    y_true, y_pred = sample_data
    mape_metric = MeanAbsolutePercentageError()
    mape_value = mape_metric(y_true, y_pred)

    expected_mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    assert np.isclose(mape_value, expected_mape), f"Expected {expected_mape}, but got {mape_value}"
    assert isinstance(mape_value, float), "MAPE value should be a float"
    assert mape_metric._name == 'Mean Absolute Percentage Error', "Metric name should be 'Mean Absolute Percentage Error'"
    assert mape_metric._description == 'Mean Absolute Percentage Error (MAPE) metric.', "Metric description should be 'Mean Absolute Percentage Error (MAPE) metric.'"


def test_accuracy_score(sample_data):
    y_true, y_pred = sample_data
    accuracy_metric = AccuracyScore()
    accuracy_value = accuracy_metric(y_true, y_pred)
    expected_accuracy = np.mean(y_true == y_pred) if len(y_true) > 0 else 0.0
    assert np.isclose(accuracy_value, expected_accuracy), f"Expected {expected_accuracy}, but got {accuracy_value}"
    assert isinstance(accuracy_value, float), "Accuracy value should be a float"
    assert accuracy_metric._name == 'Accuracy Score', "Metric name should be 'Accuracy Score'"
    assert accuracy_metric._description == 'Accuracy Score metric.', "Metric description should be 'Accuracy Score metric.'"
    
