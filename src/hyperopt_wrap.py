import random
import re

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

import optuna

class ModelHyperOpt:
    def __init__(self, train_set, eval_set, config): #TODO:need pass train and eval - in case of preprocessed (encoded, scaled) datasets
        self.train_set = train_set
        self.eval_set = eval_set
        self.config = config 
        
    def create_model(self, trial):
        
        hyperparameters = self.config["hyperparameters"](trial)
        hyperparameters.update(self.config["fixed_hyperparameters"])
        
        model = self.config['model'](**hyperparameters)
        
        return model
    
    
    
    def objective(self, trial):
        model = self.create_model(trial)
        
        X, y = self.train_set[0], self.train_set[1]  
        X_train, _, y_train, _ = train_test_split(X, y, random_state=random.randint(1, 10000), test_size=0.2)
        
        X, y = self.eval_set[0], self.eval_set[1]
        _, X_test, _, y_test = train_test_split(X, y, random_state=random.randint(1, 10000), test_size=0.8)
        
        fit_keywords_dict = {}
        for param in model.fit.__code__.co_varnames:
            if re.search('cat.*feat.*', param):
                fit_keywords_dict[param] = self.config['cat_features']
                break
        
        model.fit(X_train, y_train, **fit_keywords_dict)
            
        result = model.predict_proba(X_test)[:, 1] ## TODO: what to do if predict --> evaluate_model
        score = self.config['metric'](y_true = y_test, y_score = result)
        #TODO: need additional eval_model function
        # def evaluate_model(self, model):
        
        return score

    def process(self):
        study = optuna.create_study(
            direction = self.config['direction'],
            sampler = self.config['sampler']() if 'sampler' in self.config else None
        )
        study.optimize(self.objective, n_trials=self.config['n_trials'])
        
        best_params = study.best_params
        best_params.update(self.config["fixed_hyperparameters"])
        
        return best_params