import datetime
import os
import joblib
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
from data_splitM import X_train, Y_train, X_test, Y_test, X_all, Y_all
from reshape import reshape
from data_test import test_Xs, test_Ys
from config import PARAM_RSF, CV_TIME, CV_RATIO
config = PARAM_RSF
modelBuilder = RandomSurvivalForest
timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
model = modelBuilder(**config)
model = make_pipeline(StandardScaler(), model)
model.fit(X_train, Y_train)
surv_R_test1 = model.predict_survival_function(X_test)
surv_R_test=reshape(surv_R_test1)
surv_R_train1 = model.predict_survival_function(X_train)
surv_R_train=reshape(surv_R_train1)
prediction = model.predict(X_test)
result_TEST = concordance_index_censored(Y_test["status"], Y_test["time"], prediction)
print("RSF_TEST:")
print("c-index: {:.16}".format(result_TEST[0]))
prediction = model.predict(X_train)
result_TRAIN = concordance_index_censored(Y_train["status"], Y_train["time"], prediction)
print("RSF_TRAIN:")
print("c-index: {:.16}".format(result_TRAIN[0]))
prediction = model.predict(X_all)
result_ALL = concordance_index_censored(Y_all["status"], Y_all["time"], prediction)
print("RSF_ALL:")
print("c-index: {:.16}".format(result_ALL[0]))
surv_Rs = model.predict_survival_function(test_Xs)
predictions = model.predict(test_Xs)
result_TESTs = concordance_index_censored(test_Ys["status"], test_Ys["time"], predictions)
print("RSF_TEST:")
print("c-index: {:.16}".format(result_TESTs[0]))
surv_Rs = reshape(surv_Rs)