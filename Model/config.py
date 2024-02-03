import os

PARAM_RSF = {
    "n_estimators": 835,
    "min_samples_split": 98,
    "min_samples_leaf": 0.009336716,
    "n_jobs": os.cpu_count() - 1,
    "random_state": 666666,
}
PARAM_CPH = {
    'num_nodes': [19, 20],
    'dropout': 0.199367047938095,
    'in_features': 0,
    'out_features': 1,
    'output_bias': False,
    'batch_norm': True,
    'epoch': 1200
}

CV_TIME, CV_RATIO = 5, 1