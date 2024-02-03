import random
import numpy as np
import pandas as pd
from DataHelper import LoadData_ML

np.random.seed(666666)
train_X, train_Y = LoadData_ML(
    feature_csv="/mnt/Projects/Radiomics-CT/99.Manuscript/Data/Train/X.csv",
    surv_csv="/mnt/Projects/Radiomics-CT/99.Manuscript/Data/Train/Y.csv",
)
indices = np.arange(train_X.shape[0])
np.random.shuffle(indices)
train_X = train_X.iloc[indices]
train_Y = train_Y[indices]
x_train1 = train_X.iloc[40:50, :]
x_train2 = train_X.iloc[50:183, :]
X_train = pd.concat([x_train1, x_train2], axis=0)
X_test = train_X.iloc[0:40]
y_train1 = train_Y[40:50]
y_train2 = train_Y[50:183]
Y_train = np.concatenate([y_train1, y_train2])
Y_test = train_Y[0:40]
X_all = train_X
Y_all = train_Y