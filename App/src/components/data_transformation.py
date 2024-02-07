from src.logger import logging
from src.exception import CustomException
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.pipeline import Pipeline
import numpy as np 
import pandas as pd 
from src.utils import save_object


from dataclasses import dataclass
import sys
import os 

@dataclass
class DataTransformConfig : 
    preprocessor_path = os.path.join('artifacts','preprocessor.pkl')

class BinaryEncoder(BaseEstimator,TransformerMixin) : 
    def fit(self,x,y=None) : 
        return self 
    
    def transform(self,data) : 
        binary_encoder = {'male':0,'female':1}
        data['gender'] = [binary_encoder[i] for i in data['gender']]
        return data
    
class Encoder(BaseEstimator,TransformerMixin) : 
    def fit(self,x,y=None) :
        return self
     
    def transform(self,data) : 
        num_status = {'stunted':0,'tall':1,'normal':2,'severely stunted':3}
        data['status'] = [num_status[i] for i in data['status']]
        return data 
    
class DataTransformation() : 
    def __init__(self) : 
        self.data_transformation_config = DataTransformConfig()
        self.binary_encoder = BinaryEncoder()
        self.encoder = Encoder()

    def get_data_transformation(self) : 
        try : 
            pipe = Pipeline([
                ('binary',self.binary_encoder),
                ('enc',self.encoder)
            ])

            return pipe

        except Exception as e :
            raise CustomException(e,sys) 
        
    def initiate_data_transformation(self,train_path,test_path) : 
        try : 
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('change value of gender in english')
            train_df['gender'] = train_df['gender'].replace({'laki-laki':'male' , 'perempuan':'female'})
            test_df['gender'] = test_df['gender'].replace({'laki-laki':'male' , 'perempuan':'female'})

            logging.info('change a value of status column in english')
            train_df['status'] = train_df['status'].replace({
                'tinggi':'tall'
            })
            test_df['status'] = train_df['status'].replace({
                'tinggi':'tall'
            })

            preprocessing_obj = self.get_data_transformation()

            logging.info('applying prerocissing object on training data and testing data')

            preprocessing_obj.fit_transform(train_df)
            preprocessing_obj.fit_transform(test_df)

            logging.info('splitting data into feature and target')

            input_feature_train_arr = train_df.drop(columns=['status'],axis=1)
            target_feature_train_arr = train_df['status']

            input_feature_test_arr = test_df.drop(columns=['status'],axis=1)
            target_feature_test_arr = test_df['status']
            
            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_arr)]

            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_arr)]

            logging.info('saving preprocessing obj')
            save_object(
                file_path=self.data_transformation_config.preprocessor_path,
                obj=preprocessing_obj
            )
            logging.info(f'train_arr {train_arr}')
            logging.info(f"test_arr {test_arr}")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_path
            )
        except Exception as e : 
            raise CustomException(e,sys)





