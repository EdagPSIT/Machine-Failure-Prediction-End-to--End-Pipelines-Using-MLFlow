from machinefailure.config.core import DataProcessingConfig,DATASET_DIR,ConfigManager
import pandas as pd
from pathlib import Path
from machinefailure import logger
from sklearn.model_selection import train_test_split



class DataPreprocessing:
    def __init__(self,config=DataProcessingConfig):
        self.config =config
        self.main_processing()
    
    def load_dataset(self):
        try:
            self.df = pd.read_csv(Path(f"{DATASET_DIR}/{self.config.raw_data_path}"))
            self.df.columns = [col.replace(" ","_") for col in self.df.columns]
            logger.info('raw dataset loaded succesfuly into df.')
        except Exception as e:
            logger.error('Error occured while reading raw data into df.',e)
            raise e
        
    def drop_features(self):
        try:
            self.df.drop(self.config.features_to_drop,axis=1, inplace=True)
            logger.info('Features dropped successfuly from df and dataframe updated.')
        except Exception as e:
            logger.error('Error occured while dropping features from df.',e)
            raise e
    
    def split_save_df(self):
        try:
            df_train,df_test = train_test_split(self.df,
                                        test_size = self.config.test_size,
                                        random_state=self.config.random_state
                                        )
            df_test.to_csv(Path(f"{DATASET_DIR}/{self.config.test_data_path}"),index=False)
            df_train.to_csv(Path(f"{DATASET_DIR}/{self.config.train_data_path}"),index=False)
            logger.info('The dataset split into train and test, saved to processed directory.')
        
        except Exception as e:
            logger.error('Error occured while spliting the df into train and test')
        
    def main_processing(self):
        self.load_dataset()
        self.drop_features()
        self.split_save_df()



# if __name__ =="__main__":
#     obj = ConfigManager()
#     processing_config = obj.data_processing_config()
#     DataPreprocessing(processing_config)