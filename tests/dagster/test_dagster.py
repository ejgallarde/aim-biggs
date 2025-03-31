import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from dagster import build_op_context
from src.dagster.dags.biggs_job import (
    iris_dataset,
    split_data,
    train_model,
    predict,
    log_to_mlflow,
)


class TestIrisPipeline(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.mock_data = pd.DataFrame(
            np.random.rand(150, 4), columns=["feature1", "feature2", "feature3", "feature4"]
        )
        self.mock_data["target"] = np.random.randint(0, 3, 150)

    def test_iris_dataset(self):
        df = iris_dataset()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn("target", df.columns)

    def test_split_data(self):
        split = split_data(self.mock_data)
        self.assertIn("X_train", split)
        self.assertIn("X_test", split)
        self.assertIn("y_train", split)
        self.assertIn("y_test", split)
        self.assertEqual(len(split["X_train"]) + len(split["X_test"]), 150)

    def test_train_model(self):
        split = split_data(self.mock_data)
        model = train_model(split)
        self.assertIsInstance(model, RandomForestClassifier)

    def test_predict(self):
        split = split_data(self.mock_data)
        model = train_model(split)
        predictions = predict(model, split)
        self.assertIn("y_train_pred", predictions)
        self.assertEqual(len(predictions["y_train_pred"]), len(split["X_train"]))

    @patch("src.dagster.dags.iris_job.mlflow")
    @patch("src.dagster.dags.iris_job.plt.savefig")
    @patch("src.dagster.dags.iris_job.shap")
    @patch("src.dagster.dags.iris_job.Report")
    def test_log_to_mlflow(self, mock_report, mock_shap, mock_plt, mock_mlflow):
        mock_context = build_op_context()
        split = split_data(self.mock_data)
        model = train_model(split)
        predictions = predict(model, split)

        # Mock SHAP behavior
        mock_shap_explainer = MagicMock()
        mock_shap_explainer.shap_values.return_value = np.random.rand(len(split["X_train"]),
                                                                      len(split["X_train"].columns))
        mock_shap.TreeExplainer.return_value = mock_shap_explainer
        mock_shap.waterfall_plot.return_value = None  # Prevent actual plotting

        # Mock Evidently behavior
        mock_report_instance = MagicMock()
        mock_report.return_value = mock_report_instance

        log_to_mlflow(mock_context, model, split, predictions)

        # Verify mlflow logging
        mock_mlflow.start_run.assert_called()
        mock_mlflow.sklearn.log_model.assert_called()
        mock_mlflow.log_params.assert_called()
        mock_mlflow.log_metric.assert_called()
        mock_mlflow.log_artifact.assert_called()

        # Verify SHAP was called correctly
        mock_shap.TreeExplainer.assert_called_with(model)

        # Verify Evidently report was generated
        mock_report.assert_called()
        mock_report_instance.run.assert_called()
        mock_report_instance.save_html.assert_called_with("evidently_report.html")
        mock_mlflow.log_artifact.assert_called_with("evidently_report.html")


if __name__ == "__main__":
    unittest.main()
