# CoxPH part of the model.
import numpy as np
from pycox.evaluation import EvalSurv
from net import ATT_M
import torch
import torchtuples as tt
from pycox.models import CoxPH
from data_splitD import (
    x_train,
    x_val,
    x_test,
    y_train,
    y_test,
    y_val,
    x_trainval,
    y_trainval
)
from data_test import test_xs, test_ys
from config import PARAM_CPH, CV_TIME, CV_RATIO

_ = torch.manual_seed(666666)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
val = x_val, y_val
durations_test, events_test = y_test[0], y_test[1]
config = PARAM_CPH
config["in_features"] = x_train.shape[1]
epochs = config["epoch"]
del config["epoch"]
net = ATT_M(**config)
batch_size = 256
lr = 0.01
model = CoxPH(net, tt.optim.Adam)
model.optimizer.set_lr(lr)
callbacks = [tt.callbacks.EarlyStopping()]
log = model.fit(
    x_train,
    y_train,
    batch_size,
    epochs,
    callbacks,
    False,
    val_data=val,
    val_batch_size=batch_size,
)
model.partial_log_likelihood(*val).mean()
_ = model.compute_baseline_hazards()
surv_D_test = model.predict_surv_df(x_test)
print("Deepsurv:")
ev = EvalSurv(surv_D_test, durations_test, events_test, censor_surv="km")
print("c-index: {:.16f}".format(ev.concordance_td()))
time_grid = np.linspace(durations_test.min(), durations_test.max(), 100)
print("brier_score: {:.16f}".format(ev.integrated_brier_score(time_grid)))
surv_D_train = model.predict_surv_df(x_trainval)
print("Deepsurv_Tr:")
ev = EvalSurv(surv_D_train, y_trainval[0], y_trainval[1], censor_surv="km")
print("c-index: {:.16f}".format(ev.concordance_td()))
surv_Ds = model.predict_surv_df(test_xs)