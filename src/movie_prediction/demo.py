"""
author @ kumar dahal
this code is written to test our pieplining is working fine
"""
from movie_prediction.pipeline.pipeline import Pipeline
from movie_prediction.exception import  MovieException
from movie_prediction import logging
from movie_prediction.config.configuration import Configuartion
from movie_prediction.components.data_transformation import DataTransformation
import os

def main():
    
    try:
        config_path = os.path.join("config","config.yaml")
        pipeline = Pipeline(Configuartion(config_file_path=config_path))
        pipeline.run_pipeline()
        
        logging.info("main function execution completed.")
    except Exception as e:
        logging.error(f"{e}")
        print(e)



if __name__=="__main__":
    main()

