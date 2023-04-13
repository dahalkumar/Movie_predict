"""
author @ kumar dahal
this function is written to evaluate the old and new trained model 
and works only is model is trained again on data

"""

from movie_prediction import logging
from movie_prediction.exception import MovieException
from movie_prediction.entity.config_entity import ModelEvaluationConfig
from movie_prediction.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact,ModelTrainerArtifact,ModelEvaluationArtifact
from movie_prediction.constants import *
import numpy as np
import os
import sys
from movie_prediction.utils.common import write_yaml_file, read_yaml_file, load_object,load_data
from movie_prediction.entity.model_factory import evaluate_regression_model




class ModelEvaluation:

    def __init__(self, model_evaluation_config: ModelEvaluationConfig,
                 data_ingestion_artifact: DataIngestionArtifact, #train and test data
                 data_validation_artifact: DataValidationArtifact,#schema file 
                 model_trainer_artifact: ModelTrainerArtifact):  #train model location
        try:
            logging.info(f"{'>>' * 30}Model Evaluation log started.{'<<' * 30} ")
            self.model_evaluation_config = model_evaluation_config
            self.model_trainer_artifact = model_trainer_artifact
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact
        except Exception as e:
            raise MovieException(e, sys) from e
#there is chance when first model is run there is only one model to evaluate with existing model that is pretrained we check is 
#there  model exist in trained model path then we return model,else if we have trained model path we  read contained in dict format
    def get_best_model(self):
        try:
            model = None
            model_evaluation_file_path = self.model_evaluation_config.model_evaluation_file_path
            #check is there  model exist in trained model path then we return None
            if not os.path.exists(model_evaluation_file_path):
                write_yaml_file(file_path=model_evaluation_file_path,
                                )
                return model
            model_eval_file_content = read_yaml_file(file_path=model_evaluation_file_path)
            #model_eval_file_content is Nine then nothing else dict()
            model_eval_file_content = dict() if model_eval_file_content is None else model_eval_file_content
            #we create best_model key in the yaml file then check if it present then return model which is None
            if BEST_MODEL_KEY not in model_eval_file_content:
                return model

            model = load_object(file_path=model_eval_file_content[BEST_MODEL_KEY][MODEL_PATH_KEY])
            return model
        except Exception as e:
            raise MovieException(e, sys) from e

    def update_evaluation_report(self, model_evaluation_artifact: ModelEvaluationArtifact):
        #there could be the  scene when model_evaluation doesnot exist we create empty dict 
        #else we update model_evaluation content
        try:
            eval_file_path = self.model_evaluation_config.model_evaluation_file_path
            model_eval_content = read_yaml_file(file_path=eval_file_path)
            """
            dict() file
            """
            model_eval_content = dict() if model_eval_content is None else model_eval_content
            
            #no previous best model exist
            previous_best_model = None
            #first we check is best_model key exist 
            if BEST_MODEL_KEY in model_eval_content:
                #update the previous_best_model and  add dict of best_model
                previous_best_model = model_eval_content[BEST_MODEL_KEY]

            logging.info(f"Previous eval result: {model_eval_content}")
            #prepare new evaluation result 
            """
            best_model:
             model_path: /path/to/the/model 
            """
            eval_result = {
                BEST_MODEL_KEY: {
                    MODEL_PATH_KEY: model_evaluation_artifact.evaluated_model_path,
                }
            }
            #but is previoius_best_model exist then
            if previous_best_model is not None:
                #creating the HISTORY key in the model_evaluation_file_path, with trigger as time stamp when it was trained.
                #and previous model will be hostory
                """
                history:
                 timestamp:
                   model_path:/path/to/the/model/
                """
                model_history = {self.model_evaluation_config.time_stamp: previous_best_model}
                if HISTORY_KEY not in model_eval_content:
                    history = {HISTORY_KEY: model_history}
                    #since we already eval_result update it because previous_best_model is not None
                    eval_result.update(history)
                else:
                    model_eval_content[HISTORY_KEY].update(model_history)
            #before previous_best_model is not none wee have to update the model_evaluation_file_path and add the scenario
            #so for that we update the dict with the eval_result
            model_eval_content.update(eval_result)
            """
            now dict() file is 
            best_model:
             model_path: /path/to/the/model/with_scores   

             a = dict()

             new_a = {'best_model':'model_path: /path/to/the/model/'} 

             a.update(new_a)
            """
            logging.info(f"Updated eval result:{model_eval_content}")
            write_yaml_file(file_path=eval_file_path, data=model_eval_content)

        except Exception as e:
            raise MovieException(e, sys) from e

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            #perform the comparion of model
            #for that we trained model file path 
            trained_model_file_path = self.model_trainer_artifact.trained_model_file_path
            trained_model_object = load_object(file_path=trained_model_file_path)
            #train and test file path we have only train and test split
            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path
            #schema file 
            schema_file_path = self.data_validation_artifact.schema_file_path
            #loading npz data for train and test
            train_dataframe = load_data(file_path=train_file_path,
                                                           schema_file_path=schema_file_path,
                                                           )
            test_dataframe = load_data(file_path=test_file_path,
                                                          schema_file_path=schema_file_path,
                                                          )
            #schema file from config
            schema_content = read_yaml_file(file_path=schema_file_path)
            #get target file from schema 
            target_column_name = schema_content[TARGET_COLUMN_KEY]

            # target_column
            logging.info(f"Converting target column into numpy array.")
            train_target_arr = np.array(train_dataframe[target_column_name])
            test_target_arr = np.array(test_dataframe[target_column_name])
            logging.info(f"Conversion completed target column into numpy array.")

            # dropping target column from the dataframe
            logging.info(f"Dropping target column from the dataframe.")
            train_dataframe.drop(target_column_name, axis=1, inplace=True)
            test_dataframe.drop(target_column_name, axis=1, inplace=True)
            logging.info(f"Dropping target column from the dataframe completed.")
            #get best model
            model = self.get_best_model()

           #if no existing model then we store the trained model as our model
            if model is None:
                logging.info("Not found any existing model. Hence accepting trained model")
                #add this to artifact
                model_evaluation_artifact = ModelEvaluationArtifact(evaluated_model_path=trained_model_file_path,
                                                                    is_model_accepted=True)
                self.update_evaluation_report(model_evaluation_artifact)

                logging.info(f"Model accepted. Model eval artifact {model_evaluation_artifact} created")
                return model_evaluation_artifact
            
            #when not None  we have old model and new model create the list
            model_list = [model, trained_model_object]

            #evaluate_regression_model will evaluate our old model with new model
            metric_info_artifact = evaluate_regression_model(model_list=model_list,
                                                               X_train=train_dataframe,
                                                               y_train=train_target_arr,
                                                               X_test=test_dataframe,
                                                               y_test=test_target_arr,
                                                               base_accuracy=self.model_trainer_artifact.model_accuracy,
                                                               )
            logging.info(f"Model evaluation completed. model metric artifact: {metric_info_artifact}")
            #but if non of the metrics has a base_accuracy which means metric_info_artifact = None 
            if metric_info_artifact is None:
                #then we won't accept the model and write in ModelEvaluationArtifact
                response = ModelEvaluationArtifact(is_model_accepted=False,
                                                   evaluated_model_path=trained_model_file_path
                                                   )
                logging.info(response)
                return response

            """
            model_list = [model, trained_model_object]

            model = index_numnber = 0 old model is best
            trained_model_object = index_number =1 trained model is best
            """
            if metric_info_artifact.index_number == 1:
                model_evaluation_artifact = ModelEvaluationArtifact(evaluated_model_path=trained_model_file_path,
                                                                    is_model_accepted=True)
                self.update_evaluation_report(model_evaluation_artifact)
                logging.info(f"Model accepted. Model eval artifact {model_evaluation_artifact} created")

            else:
                logging.info("Trained model is no better than existing model hence not accepting trained model")
                model_evaluation_artifact = ModelEvaluationArtifact(evaluated_model_path=trained_model_file_path,
                                                                    is_model_accepted=False)
            return model_evaluation_artifact
        except Exception as e:
            raise MovieException(e, sys) from e

    def __del__(self):
        logging.info(f"{'=' * 20}Model Evaluation log completed.{'=' * 20} ")