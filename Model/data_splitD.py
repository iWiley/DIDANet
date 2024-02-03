import random
import numpy as np
import pandas as pd
from DataHelper import LoadData_DL
from tools import ToTuple

np.random.seed(666666)
train_x, train_y = LoadData_DL(
    feature_csv="/mnt/Projects/Radiomics-CT/99.Manuscript/Data/Train/X.csv",
    surv_csv="/mnt/Projects/Radiomics-CT/99.Manuscript/Data/Train/Y.csv",
)
indices = np.arange(train_x.shape[0])
np.random.shuffle(indices)
train_x = train_x[indices]
train_y = train_y[0][indices], train_y[1][indices]

trainx = train_x[70:183]
valx = train_x[40:70]
testx = train_x[0:40]

x_train = trainx.astype("float32")
x_val = valx.astype("float32")
x_test = testx.astype("float32")

trainy = train_y
y_train = ToTuple(70, 183, trainy)
y_val = ToTuple(40, 70, trainy)
y_test = ToTuple(0, 40, trainy)

x_trainval = train_x[40:183]
y_trainval = ToTuple(40, 183, trainy)

x_all = train_x
y_all = train_y

durations_test, events_test = y_test[0], y_test[1]
df1 = pd.DataFrame(durations_test)
df2 = pd.DataFrame(events_test)

durations_test, events_test = y_trainval[0], y_trainval[1]
df1 = pd.DataFrame(durations_test)
df2 = pd.DataFrame(events_test)

durations_test, events_test = y_all[0], y_all[1]
df1 = pd.DataFrame(durations_test)
df2 = pd.DataFrame(events_test)