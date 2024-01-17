from pathlib import Path
from typing import Dict,List,Any
from pydantic import BaseModel
from machinefailure.utils.helper_function import read_yaml
import machinefailure

# Project Directories
PACKAGE_ROOT = Path(machinefailure.__file__).resolve().parent
ROOT = PACKAGE_ROOT.parent.parent
CONFIG_FILE_PATH = ROOT / "configfiles/config.yaml"
PARAMS_FILE_PATH = ROOT / "configfiles/params.yaml"
SCHEMA_FILE_PATH = ROOT / "configfiles/schema.yaml"
DATASET_DIR = ROOT / "data"
TRAINED_MODEL_DIR = ROOT / "models"
METRIC_FILE_DIR = TRAINED_MODEL_DIR/ "model_evaluation"



class DataProcessingConfig(BaseModel):
    raw_data_path: Path
    train_data_path: Path
    test_data_path: Path
    features_to_drop: List[str]
    test_size: float
    random_state: int

class DataTransformationConfig(BaseModel):
    train_data_path: Path
    log_transform_vars: List[str]
    scale_vars: List[str]
    cat_vars: List[str]


class ModelTrainerConfig(BaseModel):
    train_data_path: Path
    save_model_name: str
    target_var: str


class ModelEvaluationConfig(BaseModel):
    test_data_path: Path
    model_path: str
    metric_file_name: str
    target_var: str
    all_params: Dict

class ConfigManager:
    def __init__(
            self,
            config_filepath = CONFIG_FILE_PATH,
            params_filepath = PARAMS_FILE_PATH,
            schema_filepath = SCHEMA_FILE_PATH
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        # self.schema = read_yaml(schema_filepath)

    def data_processing_config(self) -> DataProcessingConfig:
        config = self.config.data_processing
        params = self.params.data_processing_params

        processing_config = DataProcessingConfig(
            raw_data_path=config.raw_data_path,
            train_data_path=config.train_data_path,
            test_data_path=config.test_data_path, 
            features_to_drop=config.features_to_drop,
            test_size = params.test_size,
            random_state=params.random_state

        )
        return processing_config
    
    def data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation

        transformation_config = DataTransformationConfig(
            train_data_path=config.train_data_path,
            log_transform_vars=config.log_transform_vars,
            scale_vars=config.scale_vars,
            cat_vars=config.cat_vars
        )
        return transformation_config
    
    def trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer

        trainer_config = ModelTrainerConfig(
            train_data_path=config.train_data_path,
            save_model_name=config.save_model_name,
            target_var=config.target_var
        )
        return trainer_config
    
    def evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        params = self.params.data_processing_params

        evaluation_config = ModelEvaluationConfig(
            test_data_path=config.test_data_path,
            model_path=config.model_path,
            metric_file_name=config.metric_file_name,
            target_var=config.target_var,
            all_params=params
        )
        return evaluation_config