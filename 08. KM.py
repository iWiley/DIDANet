# This file is used to plot KM curves.
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sksurv.nonparametric import kaplan_meier_estimator
import sksurv.compare
plt.rcParams['font.sans-serif'] = "Times New Roman"
AllModels = ["DIDANet", "DeepHit", "DeepSurv", "N_MTLR", "RSF", "LASSO", "Elastic"]
Prediction_Folder = '/mnt/Projects/Radiomics-CT/99.Manuscript/Result'
df_Inst1_Y = pd.read_csv(f'/mnt/Projects/Radiomics-CT/99.Manuscript/Data/Inst.1.Y.csv', header=0)
df_OtherInst_Y = pd.read_csv(f'/mnt/Projects/Radiomics-CT/99.Manuscript/Data/Inst.Other.Y.csv', header=0)
def GetData(model, inst):
    predict = f"{Prediction_Folder}/{model}/{inst}"
    predict = pd.read_csv(predict, header = 0, index_col=0)
    predict = predict.mean().values
    return ["low" if i <= np.median(predict) else 'high' for i in predict]
def Plot_KM(time, event, group, is_incidence, cut_time = 2):
    time = time / 30
    cut_time = cut_time * 12
    time = pd.Series(time).astype(float)
    event = pd.Series(event).astype(bool)
    group = pd.Series(group).astype(str)
    if cut_time != 0:
        event = event & (time <= cut_time)
        time = time.clip(upper=cut_time)
    y = sksurv.util.Surv.from_arrays(event, time)
    _, p = sksurv.compare.compare_survival(y, group)
    all_gp = group.unique()
    all_gp.sort()
    fig = plt.figure(figsize=(5, 5))
    gs = fig.add_gridspec(10, 1)
    ax_km = fig.add_subplot(gs[:8, 0])
    ax_list = fig.add_subplot(gs[9:, 0], sharex=ax_km)
    ax_list.set_xlabel("time (month)", fontdict={'weight': 'bold'})
    ax_km.set_ylabel("est. probability of survival $\hat{S}(t)$", fontdict={'weight': 'bold'})
    ax_km.set_xlim(0, time.max())
    for value in all_gp:
        mask = group == value
        time_cell, survival_prob_cell, conf_int = kaplan_meier_estimator(
            event[mask], time[mask], conf_type="log-log"
        )
        if is_incidence:
            survival_prob_cell = 1 - survival_prob_cell
            conf_int = 1 - conf_int
            time_cell = np.insert(time_cell, 0, 0)
            survival_prob_cell = np.insert(survival_prob_cell, 0, 0)
            conf_int = np.insert(conf_int, 0, [1, 1], axis=1)
        else:
            time_cell = np.insert(time_cell, 0, 0)
            survival_prob_cell = np.insert(survival_prob_cell, 0, 1)
            conf_int = np.insert(conf_int, 0, [1, 1], axis=1)
            
        ax_km.step(
            time_cell,
            survival_prob_cell,
            where="post",
            label=f"{value} (n = {mask.sum()})",
        )
        ax_km.fill_between(time_cell, conf_int[0], conf_int[1], alpha=0.25, step="post")
        xlims = ax_km.get_xticks()
        for xl in xlims:
            index = np.abs(time_cell - xl).argmin()
            survival_rate = survival_prob_cell[index]
    if is_incidence:
        ax_km.text(
            time.max(),
            1,
            "log-rank: $p$ < 0.001" if p < 0.001 else f"log-rank: $p$ = {'%.3f'% p}",
            verticalalignment = "top", 
            horizontalalignment="right",
            fontdict={"weight": "bold"},
        )
        ax_km.set_ylim(-.05, 1)
    else:
        ax_km.text(
            time.max(),
            0,
            "log-rank: $p$ < 0.001" if p < 0.001 else f"log-rank: $p$ = {'%.3f'% p}",
            verticalalignment = "bottom", 
            horizontalalignment="right",
            fontdict={"weight": "bold"},
        )
        ax_km.set_ylim(0, 1.05)

    ax_km.set_xticks(np.arange(0, time.max() + 1, time.max() / 8).astype(int))
    ax_km.tick_params(axis="y", which="both", pad=2)
    if is_incidence:
        ax_km.legend(loc="upper left")
    else:
        ax_km.legend(loc="lower left")
    ax_list.spines["top"].set_visible(False)
    ax_list.spines["right"].set_visible(False)
    ax_list.spines["bottom"].set_visible(False)
    ax_list.spines["left"].set_visible(False)
    yt = list(range(1, len(all_gp) + 1))
    yt = np.array(yt) + 0.1
    ax_list.set_yticks(yt, reversed(all_gp))
    ax_list.tick_params(axis="y", length=0, pad=8)
    ax_y = 0
    for value in reversed(all_gp):
        ax_y += 1
        mask = group == value
        for ax_x in ax_km.get_xticks(minor=False):
            if ax_x < 0:
                continue
            if ax_x > time.max():
                continue
            n = (time[mask] >= ax_x).sum()
            if is_incidence:
                n = mask.sum() - n
            ax_list.text(
                ax_x,
                ax_y,
                0 if n < 1 else n,
                horizontalalignment="center",
                verticalalignment="center",
            )
    return fig
for m in AllModels:
    G_i1 = GetData(m, "Inst.1.csv")
    G_io = GetData(m, "Inst.Other.csv")
    T_i1 = df_Inst1_Y['T'].values
    T_io = df_OtherInst_Y['T'].values
    E_i1 = df_Inst1_Y['E'].values
    E_io = df_OtherInst_Y['E'].values
    if m == "Ours":
        km_i1 = Plot_KM(T_i1, E_i1, G_i1, False)
        km_io = Plot_KM(T_io, E_io, G_io, False)
        km_i1.savefig("/mnt/Projects/Radiomics-CT/99.Manuscript/Plots/KM.Inst.1.pdf", bbox_inches='tight')
        km_io.savefig("/mnt/Projects/Radiomics-CT/99.Manuscript/Plots/KM.Inst.Other.pdf", bbox_inches='tight')
        inc_i1 = Plot_KM(T_i1, E_i1, G_i1, True)
        inc_io = Plot_KM(T_io, E_io, G_io, True)
        inc_i1.savefig("/mnt/Projects/Radiomics-CT/99.Manuscript/Plots/INC.Inst.1.pdf", bbox_inches='tight')
        inc_io.savefig("/mnt/Projects/Radiomics-CT/99.Manuscript/Plots/INC.Inst.Other.pdf", bbox_inches='tight')
G_i1 = GetData("Ours", "Inst.1.csv")
km_tace_1 = Plot_KM(df_Inst1_Y["T"], df_Inst1_Y["E"], df_Inst1_Y["TACE"], False)
df_H_1 = df_Inst1_Y[[i == "high" for i in G_i1]]
df_L_1 = df_Inst1_Y[[i != "high" for i in G_i1]]
km_tace_H1 = Plot_KM(df_H_1["T"], df_H_1["E"], df_H_1["TACE"], False)
km_tace_L1 = Plot_KM(df_L_1["T"], df_L_1["E"], df_L_1["TACE"], False)
km_tace_1.savefig("/mnt/Projects/Radiomics-CT/99.Manuscript/Plots/KM.TACE.Inst.1.pdf", bbox_inches='tight')
km_tace_H1.savefig("/mnt/Projects/Radiomics-CT/99.Manuscript/Plots/KM.TACE.HR.Inst.1.pdf", bbox_inches='tight')
km_tace_L1.savefig("/mnt/Projects/Radiomics-CT/99.Manuscript/Plots/KM.TACE.LR.Inst.1.pdf", bbox_inches='tight')
G_io = GetData("Ours", "Inst.Other.csv")
km_tace_o = Plot_KM(df_OtherInst_Y["T"], df_OtherInst_Y["E"], df_OtherInst_Y["TACE"], False)
df_H_o = df_OtherInst_Y[[i == "high" for i in G_io]]
df_L_o = df_OtherInst_Y[[i != "high" for i in G_io]]
km_tace_Ho = Plot_KM(df_H_o["T"], df_H_o["E"], df_H_o["TACE"], False)
km_tace_Lo = Plot_KM(df_L_o["T"], df_L_o["E"], df_L_o["TACE"], False)
km_tace_o.savefig("/mnt/Projects/Radiomics-CT/99.Manuscript/Plots/KM.TACE.Inst.Other.pdf", bbox_inches='tight')
km_tace_Ho.savefig("/mnt/Projects/Radiomics-CT/99.Manuscript/Plots/KM.TACE.HR.Inst.Other.pdf", bbox_inches='tight')
km_tace_Lo.savefig("/mnt/Projects/Radiomics-CT/99.Manuscript/Plots/KM.TACE.LR.Inst.Other.pdf", bbox_inches='tight')