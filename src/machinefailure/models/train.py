import pandas as pd
from pathlib import Path
import joblib
from machinefailure import logger
from sklearn.pipeline import Pipeline
from machinefailure.config.core import ModelTrainerConfig,ConfigManager,DATASET_DIR,TRAINED_MODEL_DIR
from machinefailure.features.transformation import DataTransformation

class ModelTraining:
    def __init__(self,pipe_object:Pipeline,config=ModelTrainerConfig):
        self.config = config
        self.pipe_object = pipe_object
        self.train_model()
    def train_model(self):
        try:
            train_df = pd.read_csv(Path(f"{DATASET_DIR}/{self.config.train_data_path}"))
            X_train = train_df.drop(self.config.target_var,axis=1)
            y_train = train_df[self.config.target_var]
            self.pipe_object.fit(X_train,y_train)
            # save model to models directory
            joblib.dump(self.pipe_object,filename=f"{TRAINED_MODEL_DIR}/{self.config.save_model_name}")
            logger.info('Trained model and saved model at models directory')
        except Exception as e:
            logger.error('Error occured while training model')
            raise e