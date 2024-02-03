import numpy as np
import pandas as pd
from RSF import *
from CPH import *
from pycox.evaluation import EvalSurv
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt

def Fusion(rsf: pd.DataFrame, dps: pd.DataFrame, a):
    b = 1 - a
    rsf = rsf.loc[dps.index]
    rsf_scaled = ((rsf.T - rsf.min(axis=1)) / (rsf.max(axis=1) - rsf.min(axis=1))).T
    dps_scaled = ((dps.T - dps.min(axis=1)) / (dps.max(axis=1) - dps.min(axis=1))).T
    rsf_scaled.columns = dps.columns.values
    return a * rsf_scaled + b * dps_scaled

def Begin():
    a = 0.499936943319701

    R1 = surv_R_train
    D1 = surv_D_train
    F1 = Fusion(R1, D1, a)
    ev = EvalSurv(F1, y_trainval[0], y_trainval[1], censor_surv="km")
    print("F_TRAIN:")
    print("c-index: {:.16f}".format(ev.concordance_td()))
    time_grid = np.linspace(y_trainval[0].min(), y_trainval[0].max(), 100)
    print("brier_score: {:.16f}".format(ev.integrated_brier_score(time_grid)))
    R2 = surv_R_test
    D2 = surv_D_test
    F2 = Fusion(R2, D2, a)
    ev = EvalSurv(F2, y_test[0], y_test[1], censor_surv="km")
    print("F_TEST:")
    print("c-index: {:.16f}".format(ev.concordance_td()))
    print("brier_score: {:.16f}".format(ev.integrated_brier_score(time_grid)))
    dt = pd.concat((F2, F1), axis=1)
    y_all = (np.append(y_test[0], y_trainval[0]), np.append(y_test[1], y_trainval[1]))
    ev = EvalSurv(dt, y_all[0], y_all[1], censor_surv="km")
    print("F_INST1:")
    print("c-index: {:.16f}".format(ev.concordance_td()))
    print("brier_score: {:.16f}".format(ev.integrated_brier_score(time_grid)))

    dt.to_csv("/mnt/Projects/Radiomics-CT/99.Manuscript/Result/Ours/Inst.1.csv")

    O = Fusion(surv_Rs, surv_Ds, a)
    ev = EvalSurv(O, test_ys[0], test_ys[1], censor_surv="km")
    print("F_INST_OTHER:")
    print("c-index: {:.16f}".format(ev.concordance_td()))
    print("brier_score: {:.16f}".format(ev.integrated_brier_score(time_grid)))

    O.to_csv("/mnt/Projects/Radiomics-CT/99.Manuscript/Result/Ours/Inst.Other.csv")