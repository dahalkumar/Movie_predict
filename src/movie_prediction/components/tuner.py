
from movie_prediction.exception import MovieException
import sys
from movie_prediction import logging
from typing import List


from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import RandomizedSearchCV

from movie_prediction.constants import *


class MyModel:
    def __init__(self):
        self.rd=RandomForestRegressor()
      
    def define_the_model(self,x_train,y_train):
        """
        Parameters
        ----------
        train_x
        train_y
        Returns the model with best parameters
        -------
        """
        logging.info('Entered the define_the_model method of the MyModel class')
        try:
            self.param_grid = {"n_estimators": N_ESTIMATORS,
                               "max_depth": MAX_DEPTH,
                               "min_samples_split": MIN_SAMPLES_SPLIT,
                               "min_samples_leaf": MIN_SAMPLES_LEAF,
                               "bootstrap":BOOTSTRAP
                               }

            # Creating an object of the Grid Search class
            rf_random = RandomizedSearchCV(estimator = self.rd, param_distributions = self.param_grid,scoring='neg_mean_squared_error', n_iter = 100, cv = 5, verbose=2, random_state=42, n_jobs = 1)
            rf_random.fit(x_train,y_train)

            self.randomforest = RandomForestRegressor(**rf_random.best_params_)

    
            logging.info(f"complete Model Trainer")

            return self.randomforest
        except Exception as e:
            raise MovieException(e, sys) from e
            
    


# class ModelTuner:
#     def __init__(self):
#         self.rd=RandomForestRegressor()
#         self.gd=GradientBoostingRegressor()
        
#         self.params = read_yaml_file(file_path=MODDEL_PARAMS_PATH)
        
#     def get_best_params_for_rdf(self,train_x,train_y):
#         """
#         Parameters
#         ----------
#         train_x
#         train_y
#         Returns the model with best parameters
#         -------
#         """
#         logging.info('Entered the get_best_params_for_svm method of the Model_Finder class')
#         try:
      

#             self.param_grid = {"n_estimators": self.params.N_ESTIMATORS,
#                                "max_depth": self.params.MAX_DEPTH,
#                                "min_samples_split": self.params.MIN_SAMPLES_SPLIT,
#                                "min_samples_leaf": self.params.MIN_SAMPLES_LEAF,
#                                "bootstrap":self.params.BOOTSTRAP
#                                }

#             # Creating an object of the Grid Search class
#             self.grid = RandomizedSearchCV(estimator=self.rd,param_distributions = self.param_grid, cv=5,random_state=35, verbose=2, n_jobs=-1)

#             # finding the best parameters
#             self.grid.fit(train_x, train_y)  
#             self.n_estimators = self.grid.best_params_['n_estimators']
#             self.max_depth = self.grid.best_params_['max_depth']
#             self.min_samples_split = self.grid.best_params_['min_samples_split']
#             self.min_samples_leaf = self.grid.best_params_['min_samples_leaf']
#             self.bootstrap = self.grid.best_params_['bootstrap']

#             self.randomforest = RandomForestRegressor(n_estimators=self.n_estimators, max_depth=self.max_depth,
#                                                       min_samples_split=self.min_samples_split,min_samples_leaf=self.min_samples_leaf,bootstrap=self.bootstrap)
#             # training the mew model

            

#             logging.log('Random Foresr best params: ' + str(self.grid.best_params_) + '. Exited the get_best_params_for_rdf method of the Model_Finder class')

#             return self.randomforest.fit(train_x, train_y)
#         except MovieException as e:
#             logging.log('Exception occured in get_best_params_for_rdf method of the Model_Finder class. Exception message:  ' + str(
#                                        e))
#             logging.log('Random forest training  failed. Exited the get_best_params_for_rdf method of the Model_Finder class')
#             return e
            

    
#     def get_best_params_for_gb(self,train_x,train_y):
#             """
#             Parameters
#             ----------
#             train_x
#             train_y
#             Returns the best peremeter for gradient boosting algorithm
#             -------
#             """

#             logging.info("Entered the get_best_params_for_gb method of the Model_Finder class")
#             try:
#                 # initializing with different combination of parameters
#                 self.param_grid_xb = {
#                     "n_estimators": [100, 130],
#                     "min_samples_leaf": range(9, 10, 1),
#                     "max_depth": range(8, 10, 1),
#                     "max_leaf_nodes":range(3,9,1)

#                 }
#                 # Creating an object of the Grid Search class
#                 self.grid= GridSearchCV(estimator=self.gd, param_grid = self.param_grid_xb, cv = 2,verbose=3, n_jobs=-1)
#                 # finding the best parameters
#                 self.grid.fit(train_x, train_y)

#                 # extracting the best parameters
#                 self.n_estimators = self.grid.best_params_['n_estimators']
#                 self.max_depth = self.grid.best_params_['max_depth']
#                 self.min_samples_leaf = self.grid.best_params_['min_samples_leaf']
#                 self.max_leaf_nodes = self.grid.best_params_['max_leaf_nodes']

#                 # creating a new model with the best parameters
#                 self.xb = GradientBoostingRegressor(n_estimators=self.n_estimators, max_depth=self.max_depth,min_samples_leaf= self.min_samples_leaf,max_leaf_nodes= self.max_leaf_nodes )
                
#                 # training the mew model
                
#                 logging.info('gradient boosting: ' + str(
#                                         self.grid.best_params_) + '. Exited the get_best_params_for_gb method of the Model_Finder class')
#                 return self.xb.fit(train_x, train_y)
#             except MovieException as e:
#                 logging.info('Exception occured in get_best_params_for_gb method of the Model_Finder class. Exception message:  ' + str(
#                                         e))
#                 logging.info('XB Parameter tuning  failed. Exited the get_best_params_for_gb method of the Model_Finder class')
#                 raise Exception()


#     def get_best_model(self,train_x,train_y,test_x,test_y):
#         """
#         Parameters
#         ----------
#         train_x
#         train_y
#         test_x
#         test_y
#         Returns the model that has highest accuracy score.
#         -------
#             """
#         logging.info('Entered the get_best_model method of the Model_Finder class')
#     # create best model for gradient boosting
#         try:
#             self.gb = self.get_best_params_for_gb(train_x, train_y)
            
            
#             train_scores_gb = self.gb.score(train_x, train_y)  # r2 for GBoosting
#             test_scores_gb = self.gb.score(test_x, test_y)
#             logging.info('r2_score of gradient boosting for test is {} and train {}:'.format(test_scores_gb,train_scores_gb))  # Log AUC

#                 # create best model for Random Forest
#             self.rd = self.get_best_params_for_rdf(train_x, train_y)
#             train_scores_rd = self.rd.score(train_x, train_y)  # r2 for rdoosting
#             test_scores_rd  = self.gb.score(test_x, test_y)
#             logging.info('r2_score of random forest for test is {} and train {}:'.format(test_scores_rd,train_scores_rd)) 
            

#             # comparing the two models
#             if (test_scores_rd < test_scores_gb):
#                 return 'gradientBoosting', self.xb
#             else:
#                 return 'Random forest', self.rd

#         except Exception as e:
#             logging.info('Exception occured in get_best_model method of the Model_Finder class. Exception message:  ' + str(
#                                                 e))
#             logging.info('Model Selection Failed. Exited the get_best_model method of the Model_Finder class')
#             raise Exception()
