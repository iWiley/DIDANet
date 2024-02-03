# This document is used to plot time-dependent ROAUC curves.
import sksurv.util
import pandas as pd
import numpy as np
from sksurv.metrics import cumulative_dynamic_auc
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
plt.rcParams['font.sans-serif'] = "Times New Roman"
AllModels = ["DIDANet", "DIDANet-NC", "DeepHit", "DeepSurv", "N_MTLR", "RSF", "LASSO", "Elastic"]
Prediction_Folder = '/mnt/Projects/Radiomics-CT/99.Manuscript/Result'
pd_Inst1_Y = pd.read_csv(f'/mnt/Projects/Radiomics-CT/99.Manuscript/Data/Inst.1.Y.csv', header=0)
pd_OtherInst_Y = pd.read_csv(f'/mnt/Projects/Radiomics-CT/99.Manuscript/Data/Inst.Other.Y.csv', header=0)
AllPrediction = []
for m in AllModels:
    path_Inst1 = f"{Prediction_Folder}/{m}/Inst.1.csv"
    path_OtherInst = f"{Prediction_Folder}/{m}/Inst.Other.csv"
    pd_inst1 = pd.read_csv(path_Inst1, header=0, index_col=0)
    pd_oinst = pd.read_csv(path_OtherInst, header=0, index_col=0)
    AllPrediction.append((m, pd_inst1, pd_oinst))
def AUC(T, E, risk, v_times):
    v_times = np.clip(v_times, a_min=T.min(), a_max=T.max())
    T, E, = T.copy(), E.copy()
    result_train = []
    for val in v_times:
        idx = (pd.Series(risk.index.values) - val).abs().argsort()[:1]
        result_train.append(risk.index[idx][0])
    result_train = risk.loc[result_train].T
    E[T >= 721] = False
    T[T >= 721] = 721
    y_train = sksurv.util.Surv.from_arrays(E, T)
    y_vaild = sksurv.util.Surv.from_arrays(E, T)
    auc_train, mean_auc_train = cumulative_dynamic_auc(y_train, y_vaild, -result_train, v_times)
    return auc_train, mean_auc_train
# We're going to assess the point in time, starting at 30 days and every 30 days, all the way up to two years (721 days).
V_TIMES = np.arange(90, 721, 30)
re_i1 = pd.DataFrame(columns=np.concatenate((V_TIMES.astype(str), ['mean AUC'])))
re_io = pd.DataFrame(columns=np.concatenate((V_TIMES.astype(str), ['mean AUC'])))
for (m, i1, io) in AllPrediction:
    i1.index = i1.index.astype(float)
    io.index = io.index.astype(float)
    auc_i1, mean_auc_i1 = AUC(pd_Inst1_Y['T'].values, pd_Inst1_Y['E'].values, i1, V_TIMES)
    auc_io, mean_auc_io = AUC(pd_OtherInst_Y['T'].values, pd_OtherInst_Y['E'].values, io, V_TIMES)
    re_i1.loc[m] = np.append(auc_i1, mean_auc_i1)
    re_io.loc[m] = np.append(auc_io, mean_auc_io)
def plot(df:pd.DataFrame):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1)
    for i in range(len(df)):
        y = df.drop('mean AUC', axis=1).iloc[i, :]
        x = np.arange(len(y))
        spl = make_interp_spline(x, y)
        xnew = np.linspace(x.min(), x.max(), 500)
        ynew = spl(xnew)
        ax.plot(xnew, ynew, label=f"{df.index[i]}: mean AUC = {'%.3f' % df.iloc[i]['mean AUC']}")
    ax.legend()
    ax.set_ylim((0, 1))
    ax.set_xticks(np.arange(0, 24, 3))
    ax.set_xticklabels(np.arange(3, 25, 3))
    ax.set_xlabel("time (month)", fontdict={'weight': 'bold'})
    ax.set_ylabel("Area Under the Curve (AUC)", fontdict={'weight': 'bold'})
    return fig
auc_i1 = plot(re_i1)
auc_i1.savefig("/mnt/Projects/Radiomics-CT/99.Manuscript/Plots/AUC.Inst.1.pdf", bbox_inches='tight')
auc_io = plot(re_io)
auc_io.savefig("/mnt/Projects/Radiomics-CT/99.Manuscript/Plots/AUC.Inst.Other.pdf", bbox_inches='tight')