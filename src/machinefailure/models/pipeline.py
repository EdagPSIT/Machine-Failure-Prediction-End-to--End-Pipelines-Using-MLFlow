from machinefailure.features.processing import DataPreprocessing
from machinefailure.features.transformation import DataTransformation
from machinefailure.models.train import ModelTraining
from machinefailure.models.evaluate import ModelEvaluation
from machinefailure.config.core import ConfigManager
from machinefailure import logger


class TrainPipeline:
    # def __init__(self):
    #     self.run_train_pipe()

    def __init__(self, config_manager: ConfigManager = ConfigManager()):
        self.config_manager = config_manager
        self.run_train_pipe()


    def data_processing_stage(self):
        try:
            logger.info('Data preprocessing stage started') 
            # configmanger=ConfigManager()         
            # processing_config = configmanger.data_processing_config()
            processing_config = self.config_manager.data_processing_config()
            DataPreprocessing(processing_config)
            logger.info('Data preprocessing stage completed successfuly')
        except Exception as e:
            logger.error('There is an error at data preprocessing stage: %s', e)
            raise e

    def data_transformation_stage(self):
        try:
            logger.info('Data Transformation stage started')
            transformation_config = self.config_manager.data_transformation_config()
            dt = DataTransformation(transformation_config)
            self.failure_pipe = dt.feature_transformation()
            logger.info('Data Transformation stage completed sucessfuly')
        except Exception as e:
            logger.error('There is error at data transformation stage: %s',e)
            raise e
    
    def model_training_stage(self):
        try:
            logger.info('Model Training stage started')
            trainer_config = self.config_manager.trainer_config()
            self.failure_pipe = ModelTraining(config=trainer_config,pipe_object=self.failure_pipe)
            logger.info('Model Training stage completed sucessfuly')
        except Exception as e:
            logger.error('There is error at model training stage: %s',e)
            raise e

    def model_evaluation_stage(self):
        try:
            logger.info('Model Evaluation stage started')
            eval_config = self.config_manager.evaluation_config()
            ModelEvaluation(eval_config)
            logger.info('Model Evaluation stage completed sucessfuly')
        except Exception as e:
            logger.error('There is error at model training stage: %s',e)
            raise e

    def run_train_pipe(self):
        self.data_processing_stage()
        self.data_transformation_stage()
        self.model_training_stage()
        self.model_evaluation_stage()




if __name__ == "__main__":
    TrainPipeline()