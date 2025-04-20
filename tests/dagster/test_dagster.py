import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from dagster import build_op_context
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Import the ops and assets from your pipeline module
from src.dagster.dags.biggs_job import (
    combine_biggs_data,
    split_data,
    train_model,
    predict,
    log_to_mlflow,
)

class TestBiggsPipeline(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        # Create a mock time series DataFrame with a datetime index for sequential splitting
        dates = pd.date_range(start="2023-01-01", periods=150, freq="D")
        self.mock_data = pd.DataFrame(
            np.random.rand(150, 4), columns=["feature1", "feature2", "feature3", "feature4"]
        )
        # For regression, use continuous target values
        self.mock_data["target"] = np.random.rand(150)
        self.mock_data.index = dates

    def test_biggs_dataset(self):
        # Call the asset without any inputs; biggs_dataset should now not require external_data.
        df = combine_biggs_data()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn("target", df.columns)

    def test_split_data(self):
        split = split_data(self.mock_data)
        self.assertIn("X_train", split)
        self.assertIn("X_test", split)
        self.assertIn("y_train", split)
        self.assertIn("y_test", split)
        # Ensure the split covers the entire dataset
        self.assertEqual(len(split["X_train"]) + len(split["X_test"]), len(self.mock_data))

    def test_train_model(self):
        split = split_data(self.mock_data)
        context = build_op_context()
        model_result = train_model(context, split)
        model, algo = model_result
        # Expect the model to be an instance of either XGBRegressor or LGBMRegressor
        self.assertTrue(isinstance(model, (XGBRegressor, LGBMRegressor)))

    def test_predict(self):
        split = split_data(self.mock_data)
        context = build_op_context()
        model_result = train_model(context, split)
        predictions = predict(context, model_result, split)
        # Expect predictions on test set
        self.assertIn("y_test_pred", predictions)
        self.assertEqual(len(predictions["y_test_pred"]), len(split["X_test"]))

    @patch("src.dagster.dags.biggs_job.mlflow")
    @patch("src.dagster.dags.biggs_job.plt.savefig")
    @patch("src.dagster.dags.biggs_job.shap")
    @patch("src.dagster.dags.biggs_job.Report")
    def test_log_to_mlflow(self, mock_report, mock_shap, mock_plt, mock_mlflow):
        mock_context = build_op_context()
        split = split_data(self.mock_data)
        model_result = train_model(mock_context, split)
        predictions = predict(mock_context, model_result, split)
        model, algo = model_result

        # Mock SHAP behavior
        mock_shap_explainer = MagicMock()
        # Assume the explainer returns an array with shape (n_samples, n_features)
        mock_shap_explainer.shap_values.return_value = np.random.rand(len(split["X_train"]), split["X_train"].shape[1])
        mock_shap.TreeExplainer.return_value = mock_shap_explainer
        # Prevent actual plotting
        mock_shap.waterfall_plot.return_value = None

        # Mock Evidently behavior
        mock_report_instance = MagicMock()
        mock_report.return_value = mock_report_instance

        log_to_mlflow(mock_context, model_result, split, predictions)

        # Verify mlflow logging calls
        mock_mlflow.start_run.assert_called()
        mock_mlflow.sklearn.log_model.assert_called()
        mock_mlflow.log_params.assert_called()
        mock_mlflow.log_metric.assert_called()
        mock_mlflow.log_artifact.assert_called()

        # Verify SHAP was used correctly
        mock_shap.TreeExplainer.assert_called_with(model)

        # Verify Evidently report was generated
        mock_report.assert_called()
        mock_report_instance.run.assert_called()
        mock_report_instance.save_html.assert_called_with("evidently_report.html")
        mock_mlflow.log_artifact.assert_called_with("evidently_report.html")

if __name__ == "__main__":
    unittest.main()
