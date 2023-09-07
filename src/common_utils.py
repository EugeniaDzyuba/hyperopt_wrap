import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from category_encoders.target_encoder import TargetEncoder
from sklearn.linear_model import LogisticRegression

def prc_auc(
    y_true: np.ndarray, 
    y_score: np.ndarray
) -> float:
    pr, rec, _ = metrics.precision_recall_curve(
        y_true=y_true,
        probas_pred=y_score
    )
    
    pr_auc = round(
        metrics.auc(
            x=rec,
            y=pr
        ), 3
    )
    
    return pr_auc


def preprocess(
    X: pd.DataFrame,
    features_names: list,
    cat_features: list,
    num_features: list,
    y: np.ndarray = None,
    num_features_means: pd.Series = None,
    encoder = None,
    scaler = None,
    return_obj: bool = False
):
    #fillna

    if num_features_means is None:
        num_features_means = X[num_features].mean()

    X[num_features] = X[num_features].fillna(num_features_means)

    #encode
    if encoder is None:
        encoder = TargetEncoder()
        encoder.fit(X = X[cat_features], y = y)

    encoded_cat_features = encoder.transform(X[cat_features])
    X_encoded = pd.concat((encoded_cat_features, X[num_features]), axis = 1)

    #scale
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(X_encoded[features_names].values)

    X_prepared = scaler.transform(X_encoded[features_names])

    if return_obj:
        return {
            'X' : X_prepared,
            'num_means' : num_features_means,
            'encoder' : encoder,
            'scaler' : scaler
        }
    else:
        return X_prepared

def train_model(
    X_prepared: np.ndarray,
    y: np.ndarray,
    features_names: list,
    cat_features: list,
    num_features: list,
    params: dict
):
    model = LogisticRegression(**params)
    model.fit(X_prepared, y)

    return model

def preprocess_and_train(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    features_names: list,
    cat_features: list,
    num_features: list,
    params: dict = {}
):

    return_dict = preprocess(
        X = X_train,
        y = y_train,
        features_names = features_names,
        cat_features = cat_features,
        num_features = num_features,
        return_obj = True
    )

    X_preprocessed = return_dict['X']
    return_dict.pop('X', None)

    return_dict['model'] = train_model(
        X_prepared = X_preprocessed,
        y = y_train,
        features_names = features_names,
        cat_features = cat_features,
        num_features = num_features,
        params = params
    )

    return return_dict 