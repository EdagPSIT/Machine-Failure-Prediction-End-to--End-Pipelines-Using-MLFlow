from machinefailure.config.core import DataTransformationConfig,DATASET_DIR,ConfigManager
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from machinefailure.features.common_transformation import LogTransformer
import joblib



class DataTransformation:
    def __init__(self,config=DataTransformationConfig) -> None:
        self.config = config
        self.feature_transformation()
    
    def feature_transformation(self):

        num_pipeline = Pipeline([
            ('Imputer',SimpleImputer(strategy='median')),
            ('Scaler',MinMaxScaler())
        ])

        num_log_transformer = Pipeline([
            ('Log Transform',LogTransformer())
        ])

        cat_pipeline = Pipeline([
            ('Imputer',SimpleImputer(strategy='most_frequent')),
            ("OHE",OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer([
            ('Numerical Transformation',num_pipeline,self.config.scale_vars),
            ('Categorical Transformation',cat_pipeline,self.config.cat_vars),
            ('Log Transformation',num_log_transformer,self.config.log_transform_vars)
        ])

        log_reg = LogisticRegression()

        failure_pipeline = ImbPipeline([
            ('Preprocessor',preprocessor),
            ('smote', SMOTE(sampling_strategy='minority', random_state=42)), # SMOTE for oversampling minority classes
            ('Logistic Regression',log_reg)
        ])

        # saving pipeline
        return failure_pipeline