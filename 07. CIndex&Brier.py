# This document is used to plot time-dependent C-index curves as well as time-dependent Brier score curves.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pycox.evaluation import EvalSurv
from sksurv.metrics import concordance_index_censored
from scipy.interpolate import make_interp_spline
from matplotlib.colors import LinearSegmentedColormap
plt.rcParams['font.sans-serif'] = "Times New Roman"
AllModels = ["DIDANet", "DIDANet-NC", "DeepHit", "DeepSurv", "N_MTLR", "RSF", "LASSO", "Elastic"]
Prediction_Folder = '/mnt/Projects/Radiomics-CT/99.Manuscript/Result'
pd_Inst1_Y = pd.read_csv(f'/mnt/Projects/Radiomics-CT/99.Manuscript/Data/Inst.1.Y.csv', header=0)
pd_OtherInst_Y = pd.read_csv(f'/mnt/Projects/Radiomics-CT/99.Manuscript/Data/Inst.Other.Y.csv', header=0)
inst2 = 28
inst3 = 35 + inst2
inst4 = 37 + inst3
pd_Inst2_Y = pd_OtherInst_Y.iloc[:inst2]
pd_Inst3_Y = pd_OtherInst_Y.iloc[inst2:inst3]
pd_Inst4_Y = pd_OtherInst_Y.iloc[inst3:inst4]
pd_Inst5_Y = pd_OtherInst_Y.iloc[inst4:]    
AllPrediction = []
for m in AllModels:
    path_Inst1 = f"{Prediction_Folder}/{m}/Inst.1.csv"
    path_OtherInst = f"{Prediction_Folder}/{m}/Inst.Other.csv"
    pd_inst1 = pd.read_csv(path_Inst1, header=0, index_col=0)
    pd_oinst = pd.read_csv(path_OtherInst, header=0, index_col=0)
    pd_inst2 = pd_oinst.iloc[:, :inst2]
    pd_inst3 = pd_oinst.iloc[:, inst2:inst3]
    pd_inst4 = pd_oinst.iloc[:, inst3:inst4]
    pd_inst5 = pd_oinst.iloc[:, inst4:]    
    AllPrediction.append((m, pd_inst1, pd_oinst, pd_inst2, pd_inst3, pd_inst4, pd_inst5))
def Calc_TimeCIndex(time, event, risk, timepoint):
    cindex = []
    idx = risk.index[(np.abs(risk.index.values[:, None] - timepoint)).argmin(axis=0)]
    new_df = risk.loc[idx]
    event[time>=721] = False
    time[time>=721] = 721
    ev = EvalSurv(risk, time, event, censor_surv='km')
    for i in range(0, len(timepoint)):
        c = concordance_index_censored(event.astype('bool'), time, 1-new_df.iloc[i])
        cindex.append(c[0])
    time_cindex = pd.DataFrame(
        { 'C-Index' : cindex },
        index=timepoint
    )
    return time_cindex, ev.concordance_td(), ev.brier_score(timepoint), ev.integrated_brier_score(timepoint)
# We're going to assess the point in time, starting at 30 days and every 30 days, all the way up to two years (721 days).
V_TIMES = np.arange(90, 721, 30)
re_cindex_i1 = pd.DataFrame(columns=V_TIMES.astype(str))
re_cindex_i2 = pd.DataFrame(columns=V_TIMES.astype(str))
re_cindex_i3 = pd.DataFrame(columns=V_TIMES.astype(str))
re_cindex_i4 = pd.DataFrame(columns=V_TIMES.astype(str))
re_cindex_i5 = pd.DataFrame(columns=V_TIMES.astype(str))
re_cindex_io = pd.DataFrame(columns=V_TIMES.astype(str))
re_brier_i1  = pd.DataFrame(columns=V_TIMES.astype(str))
re_brier_i2  = pd.DataFrame(columns=V_TIMES.astype(str))
re_brier_i3  = pd.DataFrame(columns=V_TIMES.astype(str))
re_brier_i4  = pd.DataFrame(columns=V_TIMES.astype(str))
re_brier_i5  = pd.DataFrame(columns=V_TIMES.astype(str))
re_brier_io  = pd.DataFrame(columns=V_TIMES.astype(str))
re_ibrier_i1  = pd.DataFrame(columns=V_TIMES.astype(str))
re_ibrier_i2  = pd.DataFrame(columns=V_TIMES.astype(str))
re_ibrier_i3  = pd.DataFrame(columns=V_TIMES.astype(str))
re_ibrier_i4  = pd.DataFrame(columns=V_TIMES.astype(str))
re_ibrier_i5  = pd.DataFrame(columns=V_TIMES.astype(str))
re_ibrier_io  = pd.DataFrame(columns=V_TIMES.astype(str))
for (m, i1, io, i2, i3, i4, i5) in AllPrediction:
    i1.index = i1.index.astype(float)
    i2.index = i2.index.astype(float)
    i3.index = i3.index.astype(float)
    i4.index = i4.index.astype(float)
    i5.index = i5.index.astype(float)
    io.index = io.index.astype(float)
    cindex_i1, mean_cindex_i1, brier_i1, ibrier_i1 = Calc_TimeCIndex(pd_Inst1_Y['T'].values, pd_Inst1_Y['E'].values, i1, V_TIMES)
    cindex_i2, mean_cindex_i2, brier_i2, ibrier_i2 = Calc_TimeCIndex(pd_Inst2_Y['T'].values, pd_Inst2_Y['E'].values, i2, V_TIMES)
    cindex_i3, mean_cindex_i3, brier_i3, ibrier_i3 = Calc_TimeCIndex(pd_Inst3_Y['T'].values, pd_Inst3_Y['E'].values, i3, V_TIMES)
    cindex_i4, mean_cindex_i4, brier_i4, ibrier_i4 = Calc_TimeCIndex(pd_Inst4_Y['T'].values, pd_Inst4_Y['E'].values, i4, V_TIMES)
    cindex_i5, mean_cindex_i5, brier_i5, ibrier_i5 = Calc_TimeCIndex(pd_Inst5_Y['T'].values, pd_Inst5_Y['E'].values, i5, V_TIMES)
    cindex_io, mean_cindex_io, brier_io, ibrier_io = Calc_TimeCIndex(pd_OtherInst_Y['T'].values, pd_OtherInst_Y['E'].values, io, V_TIMES)
    re_cindex_i1.loc[m] = cindex_i1.values.reshape(-1)
    re_cindex_i2.loc[m] = cindex_i2.values.reshape(-1)
    re_cindex_i3.loc[m] = cindex_i3.values.reshape(-1)
    re_cindex_i4.loc[m] = cindex_i4.values.reshape(-1)
    re_cindex_i5.loc[m] = cindex_i5.values.reshape(-1)
    re_cindex_io.loc[m] = cindex_io.values.reshape(-1)
    re_brier_i1.loc[m] = brier_i1.values.T
    re_brier_i2.loc[m] = brier_i2.values.T
    re_brier_i3.loc[m] = brier_i3.values.T
    re_brier_i4.loc[m] = brier_i4.values.T
    re_brier_i5.loc[m] = brier_i5.values.T
    re_brier_io.loc[m] = brier_io.values.T
    re_ibrier_i1.loc[m] = ibrier_i1
    re_ibrier_i2.loc[m] = ibrier_i2
    re_ibrier_i3.loc[m] = ibrier_i3
    re_ibrier_i4.loc[m] = ibrier_i4
    re_ibrier_i5.loc[m] = ibrier_i5
    re_ibrier_io.loc[m] = ibrier_io
def plot_time_cindex(df:pd.DataFrame):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1)
    for i in range(len(df)):
        y = df.iloc[i, :]
        x = np.arange(len(y))
        spl = make_interp_spline(x, y)
        xnew = np.linspace(x.min(), x.max(), 100)
        ynew = spl(xnew)
        ax.plot(xnew, ynew, label=f"{df.index[i]}: mean C-Index = {'%.3f' % df.iloc[i].mean()}")
    ax.legend()
    ax.set_ylim((.2, 1))
    ax.set_xticks(np.arange(0, 24, 3))
    ax.set_xticklabels(np.arange(3, 25, 3))
    ax.set_xlabel("Time (month)", fontdict={'weight': 'bold'})
    ax.set_ylabel("Concordance Index", fontdict={'weight': 'bold'})
    return fig
def plot_time_brier(df:pd.DataFrame):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1)
    for i in range(len(df)):
        y = df.iloc[i, :]
        x = np.arange(len(y))
        spl = make_interp_spline(x, y)
        xnew = np.linspace(x.min(), x.max(), 100)
        ynew = spl(xnew)
        ax.plot(xnew, ynew, label=f"{df.index[i]}: mean brier = {'%.3f' % df.iloc[i].mean()}")
    ax.legend()
    ax.set_ylim((0, .5))
    ax.set_xticks(np.arange(0, 24, 3))
    ax.set_xticklabels(np.arange(3, 25, 3))
    ax.set_xlabel("Time (month)", fontdict={'weight': 'bold'})
    ax.set_ylabel("Brier Score", fontdict={'weight': 'bold'})
    return fig
b1 = re_brier_i1.T.mean()
b2 = re_brier_i2.T.mean()
b3 = re_brier_i3.T.mean()
b4 = re_brier_i4.T.mean()
b5 = re_brier_i5.T.mean()
pd_briers = pd.concat([b1, b2, b3, b4, b5], axis=1)
pd_briers.columns = ["Institution 1", "Institution 2", "Institution 3", "Institution 4", "Institution 5"]
plt.clf()
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(1, 1, 1)
cmap = LinearSegmentedColormap.from_list("cmap", ['#1f77b4', '#ff7f0e'])
hot = ax.imshow(pd_briers, cmap=cmap)
for i in range(pd_briers.shape[0]):
    for j in range(pd_briers.shape[1]):
        text = ax.text(j, i, round(pd_briers.iloc[i, j], 3),
                       ha="center", va="center", color="k")
ax.set_xticklabels(pd_briers.columns.values)
ax.set_xticks(range(len(pd_briers.columns.values)), labels=pd_briers.columns.values, rotation=45)
ax.set_yticks(range(len(pd_briers.index.values)), labels=pd_briers.index.values)
brier_i1 = plot_time_brier(re_brier_i1)
brier_io = plot_time_brier(re_brier_io)
brier_i1.savefig("/mnt/Projects/Radiomics-CT/99.Manuscript/Plots/Brier.Inst.1.pdf", bbox_inches='tight')
brier_io.savefig("/mnt/Projects/Radiomics-CT/99.Manuscript/Plots/Brier.Inst.Other.pdf", bbox_inches='tight')
cindex_i1 = plot_time_cindex(re_cindex_i1)
cindex_io = plot_time_cindex(re_cindex_io)
cindex_i1.savefig("/mnt/Projects/Radiomics-CT/99.Manuscript/Plots/CI.Inst.1.pdf", bbox_inches='tight')
cindex_io.savefig("/mnt/Projects/Radiomics-CT/99.Manuscript/Plots/CI.Inst.Other.pdf", bbox_inches='tight')