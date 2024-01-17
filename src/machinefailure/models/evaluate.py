import os
import pandas as pd
from sklearn.metrics import precision_score,recall_score,roc_auc_score
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import numpy as np
import joblib
from pathlib import Path
from machinefailure.config.core import ModelEvaluationConfig

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    
    def eval_metrics(self,actual, pred):
        precision = precision_score(actual, pred)
        recall = recall_score(actual, pred)
        roc_auc = roc_auc_score(actual, pred)
        return precision,recall,roc_auc    


    def log_into_mlflow(self):

        test_df = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        X_test = test_df.drop([self.config.target_var], axis=1)
        y_test = test_df[[self.config.target_var]]


        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme


        with mlflow.start_run():

            predicted_qualities = model.predict(test_x)

            (rmse, mae, r2) = self.eval_metrics(test_y, predicted_qualities)
            
            # Saving metrics as local
            scores = {"rmse": rmse, "mae": mae, "r2": r2}
            save_json(path=Path(self.config.metric_file_name), data=scores)

            mlflow.log_params(self.config.all_params)

            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("mae", mae)


            # Model registry does not work with file store
            if tracking_url_type_store != "file":

                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.sklearn.log_model(model, "model", registered_model_name="ElasticnetModel")
            else:
                mlflow.sklearn.log_model(model, "model")

    