
from movie_prediction.exception import MovieException
import sys
from movie_prediction import logging
from typing import List
from movie_prediction.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from movie_prediction.entity.config_entity import ModelTrainerConfig
from movie_prediction.utils.common import load_numpy_array_data,save_object,load_object,read_yaml_file

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics  import r2_score,accuracy_score
from movie_prediction.constants import *

#from movie_prediction.components.tuner import MyModel

from typing import List

from movie_prediction.entity.model_factory import MetricInfoArtifact, ModelFactory,GridSearchedBestModel
from movie_prediction.entity.model_factory import evaluate_regression_model



class MovieModel:
    def __init__(self, preprocessing_object, trained_model_object):
        """
        TrainedModel constructor
        preprocessing_object: preprocessing_object
        trained_model_object: trained_model_object
        """
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object

    def predict(self, X):
        """
        function accepts raw inputs and then transformed raw input using preprocessing_object
        which gurantees that the inputs are in the same format as the training data
        At last it perform prediction on transformed features
        """
        transformed_feature = self.preprocessing_object.transform(X)
        return self.trained_model_object.predict(transformed_feature)

    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"

    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"




class ModelTrainer:

    def __init__(self, model_trainer_config:ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        try:
            logging.info(f"{'>>' * 30}Model trainer log started.{'<<' * 30} ")
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise MovieException(e, sys) from e

    def initiate_model_trainer(self)->ModelTrainerArtifact:
        try:
            logging.info(f"Loading transformed training dataset")
            transformed_train_file_path = self.data_transformation_artifact.transformed_train_file_path
            train_array = load_numpy_array_data(file_path=transformed_train_file_path)

            logging.info(f"Loading transformed testing dataset")
            transformed_test_file_path = self.data_transformation_artifact.transformed_test_file_path
            test_array = load_numpy_array_data(file_path=transformed_test_file_path)

            logging.info(f"Splitting training and testing input and target feature")
            x_train,y_train,x_test,y_test = train_array[:,:-1],train_array[:,-1],test_array[:,:-1],test_array[:,-1]
            

            logging.info(f"Extracting model config file path")
            model_config_file_path = self.model_trainer_config.model_config_file_path

            logging.info(f"Initializing model factory class using above model config file: {model_config_file_path}")
            model_factory = ModelFactory(model_config_path=model_config_file_path)
            
            
            base_accuracy = self.model_trainer_config.base_accuracy
            logging.info(f"Expected accuracy: {base_accuracy}")

            logging.info(f"Initiating operation model selecttion")
            best_model = model_factory.get_best_model(X=x_train,y=y_train,base_accuracy=base_accuracy)
            
            logging.info(f"Best model found on training dataset: {best_model}")
            
            logging.info(f"Extracting trained model list.")
            grid_searched_best_model_list:List[GridSearchedBestModel]=model_factory.grid_searched_best_model_list
            
            model_list = [model.best_model for model in grid_searched_best_model_list ]
            logging.info(f"Evaluation all trained model on training and testing dataset both")
            metric_info:MetricInfoArtifact = evaluate_regression_model(model_list=model_list,X_train=x_train,y_train=y_train,X_test=x_test,y_test=y_test,base_accuracy=base_accuracy)

            logging.info(f"Best found model on both training and testing dataset.")
            
            preprocessing_obj=  load_object(file_path=self.data_transformation_artifact.preprocessed_object_file_path)
            model_object = metric_info.model_object


            trained_model_file_path=self.model_trainer_config.trained_model_file_path
            movie_model = MovieModel(preprocessing_object=preprocessing_obj,trained_model_object=model_object)
            logging.info(f"Saving model at path: {trained_model_file_path}")
            save_object(file_path=trained_model_file_path,obj=movie_model)


            model_trainer_artifact=  ModelTrainerArtifact(is_trained=True,message="Model Trained successfully",
            trained_model_file_path=trained_model_file_path,
            train_rmse=metric_info.train_rmse,
            test_rmse=metric_info.test_rmse,
            train_accuracy=metric_info.train_accuracy,
            test_accuracy=metric_info.test_accuracy,
            model_accuracy=metric_info.model_accuracy
            
            )

            logging.info(f"Model Trainer Artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise MovieException(e, sys) from e

    def __del__(self):
        logging.info(f"{'>>' * 30}Model trainer log completed.{'<<' * 30} ")
# class MovieModel:
#     def __init__(self, preprocessing_object, trained_model_object):
#         """
#         TrainedModel constructor
#         preprocessing_object: preprocessing_object
#         trained_model_object: trained_model_object
#         """
#         self.preprocessing_object = preprocessing_object
#         self.trained_model_object = trained_model_object
#         self.rd=RandomForestRegressor()
#         self.params = read_yaml_file(file_path=MODDEL_PARAMS_PATH)
        
            
#     def predict(self, X):
#         """
#         function accepts raw inputs and then transformed raw input using preprocessing_object
#         which gurantees that the inputs are in the same format as the training data
#         At last it perform prediction on transformed features
#         """
#         transformed_feature = self.preprocessing_object.transform(X)
#         return self.trained_model_object.predict(transformed_feature)

#     def __repr__(self):
#         return f"{type(self.trained_model_object).__name__}()"

#     def __str__(self):
#         return f"{type(self.trained_model_object).__name__}()"




# class ModelTrainer:

#     def __init__(self, model_trainer_config:ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
#         try:
#             logging.info(f"{'>>' * 30}Model trainer log started.{'<<' * 30} ")
#             self.model_trainer_config = model_trainer_config
#             self.data_transformation_artifact = data_transformation_artifact
#         except Exception as e:
#             raise MovieException(e, sys) from e

#     def initiate_model_trainer(self)->ModelTrainerArtifact:
#         try:
#             logging.info(f"Loading transformed training dataset")
#             transformed_train_file_path = self.data_transformation_artifact.transformed_train_file_path
#             train_array = load_numpy_array_data(file_path=transformed_train_file_path)

#             logging.info(f"Loading transformed testing dataset")
#             transformed_test_file_path = self.data_transformation_artifact.transformed_test_file_path
#             test_array = load_numpy_array_data(file_path=transformed_test_file_path)

#             logging.info(f"Splitting training and testing input and target feature")
#             train_x,train_y,test_x,test_y = train_array[:,:-1],train_array[:,-1],test_array[:,:-1],test_array[:,-1]
            

#             logging.info(f"Extracting model config file path")
#             model_config_file_path = self.model_trainer_config.model_config_file_path

#             logging.info(f"Initiating operation model selection: {model_config_file_path}")
#             mymodel = MyModel()
#             model_object  = mymodel.define_the_model(train_x,train_y)
            
#             model_object.fit(train_x,train_y)

#             logging.info("Coefficient of determination R^2 <-- on train set: {}".format(model_object.score(train_x, train_y)))
#             logging.info("Coefficient of determination R^2 <-- on test set: {}".format(model_object.score(test_x, test_y)))


       
#             logging.info(f"Best found model on both training and testing dataset.")
            
#             preprocessing_obj=  load_object(file_path=self.data_transformation_artifact.preprocessed_object_file_path)
        

            
#             trained_model_file_path=self.model_trainer_config.trained_model_file_path
#             housing_model = HousingEstimatorModel(preprocessing_object=preprocessing_obj,trained_model_object=model_object)
#             logging.info(f"Saving model at path: {trained_model_file_path}")
#             save_object(file_path=trained_model_file_path,obj=housing_model)


#             model_trainer_artifact=  ModelTrainerArtifact(is_trained=True,message="Model Trained successfully",
#             trained_model_file_path=trained_model_file_path,
#             )

#             logging.info(f"Model Trainer Artifact: {model_trainer_artifact}")
#             return model_trainer_artifact
#         except Exception as e:
#             raise MovieException(e, sys) from e

#     def __del__(self):
#         logging.info(f"{'>>' * 30}Model trainer log completed.{'<<' * 30} ")



