import os 
import sys 
from src.logger import logging
from src.exception import CustomException
import pandas as pd 

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation


logging.info('config file path name')
#iniate train,test,raw data path  
@dataclass
class DataIngestionConfig : 
    train_data_path : str=os.path.join('splited_datas','train.csv')
    test_data_path : str=os.path.join('splited_datas','test.csv')
    raw_data_path : str=os.path.join('splited_datas','raw.csv')


class DataIngestion : 
    def __init__(self) : 
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self) : 
        try :
            logging.info('read csv file') 
            df = pd.read_csv('data/data_balita.csv')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            logging.info('renaming columns in english')
            df = df.rename(columns={
                "Umur (bulan)" : "age(month)",
                "Jenis Kelamin" : "gender",
                "Tinggi Badan (cm)" : "height(cm)",
                "Status Gizi" : "status"
            })


            logging.info('split data into 20% test data and train data')
            train_data,test_data = train_test_split(df,test_size=0.2,random_state=42)

            logging.info('save the data to file path got configed before')
            train_data.to_csv(self.ingestion_config.train_data_path)
            test_data.to_csv(self.ingestion_config.test_data_path)

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e : 
            raise CustomException(e,sys)


if __name__ == '__main__' : 
    obj = DataIngestion()
    train_data,test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_data,test_data)

