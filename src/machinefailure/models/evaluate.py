import os
import pandas as pd
from sklearn.metrics import precision_score,recall_score,roc_auc_score
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import numpy as np
import joblib
from pathlib import Path
from machinefailure import logger
from machinefailure.config.core import ModelEvaluationConfig,DATASET_DIR,METRIC_FILE_DIR,TRAINED_MODEL_DIR
from machinefailure.utils.helper_function import save_json

class ModelEvaluation:

    """Class for evaluating machine learning models."""

    def __init__(self, config: ModelEvaluationConfig):

        """Initialize ModelEvaluation with the given configuration."""
        self.config = config
        self.log_into_mlflow()

    
    def eval_metrics(self,actual, pred):
        """Calculate evaluation metrics for the model."""
        precision = precision_score(actual, pred)
        recall = recall_score(actual, pred)
        roc_auc = roc_auc_score(actual, pred)
        return precision,recall,roc_auc    


    def log_into_mlflow(self):
        """Log evaluation metrics and model to MLflow."""
        logger.info('Experiment logging started')
        
        test_df = pd.read_csv(Path(f"{DATASET_DIR}/{self.config.test_data_path}"))

        model_path = Path(TRAINED_MODEL_DIR, self.config.model_path)
        if model_path.exists():
            model = joblib.load(model_path)
        else:
            logger.error('Model file not found error at models directory')
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # model = joblib.load(Path(f"{TRAINED_MODEL_DIR}/{self.config.model_path}"))

        X_test = test_df.drop([self.config.target_var], axis=1)
        y_test = test_df[[self.config.target_var]]


        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        print(mlflow.set_registry_uri(self.config.mlflow_uri))

        with mlflow.start_run():

            predicted_qualities = model.predict(X_test)

            (precision,recall,roc_auc) = self.eval_metrics(y_test, predicted_qualities)
            
            # Saving metrics as local
            scores = {"precision": precision, "recall": recall, "roc_auc": roc_auc}
            save_json(path=Path(f"{METRIC_FILE_DIR}/{self.config.metric_file_name}"), data=scores)

            mlflow.log_params(self.config.all_params)

            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("roc_auc", roc_auc)
            logger.info('Metrics for this experiments logged to mlflow.')


            # Model registry does not work with file store
            if tracking_url_type_store != "file":

                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.sklearn.log_model(model, "model", registered_model_name="Logistic Regression")
            else:
                mlflow.sklearn.log_model(model, "model")
            logger.info('Experiment logging completed.')