# This file is used to plot the relevant data from the R language output.
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
plt.rcParams['font.sans-serif'] = "Times New Roman"
path_data_dir = "/mnt/Projects/Radiomics-CT/99.Manuscript/Data/Plot"
path_cal_i1 = f"{path_data_dir}/CAL.Inst.1.csv"
path_cal_io = f"{path_data_dir}/CAL.Inst.Other.csv"
path_cal_i1_Tr = f"{path_data_dir}/CAL.Inst.1.Tr.csv"
path_cal_i1_Te = f"{path_data_dir}/CAL.Inst.1.Te.csv"
path_dca_i1 = f"{path_data_dir}/DCA.Inst.1.csv"
path_dca_io = f"{path_data_dir}/DCA.Inst.Other.csv"
pd_dca_i1 = pd.read_csv(path_dca_i1, header=0)
pd_dca_io = pd.read_csv(path_dca_io, header=0)
pd_cal_i1 = pd.read_csv(path_cal_i1, header=0)
pd_cal_io = pd.read_csv(path_cal_io, header=0)
pd_cal_i1_Tr = pd.read_csv(path_cal_i1_Tr, header=0)
pd_cal_i1_Te = pd.read_csv(path_cal_i1_Te, header=0)
models = ["DeepHit", "DeepSurv", "N_MTLR", "Elastic", "LASSO", "RSF"]
cilncs = ["AJCC", "BCLC", "CLIP", "CNLC", "CUPI", "HKLC", "ITA.LI.CA", "Okuda"]
def plot_dca(df: pd.DataFrame, frac=0.2):
    fig_m = plt.figure(figsize=(5, 5))
    fig_c = plt.figure(figsize=(5, 5))
    ax_m = fig_m.add_subplot(1, 1, 1)
    ax_c = fig_c.add_subplot(1, 1, 1)
    x = df.loc[df["model"] == "DIDANet", "thresholds"].values
    y = df.loc[df["model"] == "DIDANet", "NB"].values
    lowess = sm.nonparametric.lowess(y, x, frac=frac)
    xnew = lowess[:, 0]
    ynew = lowess[:, 1]
    ax_c.plot(xnew, ynew, label="DIDANet", linewidth=2)
    ax_m.plot(xnew, ynew, label="DIDANet", linewidth=2)
    x = df.loc[df["model"] == "All", "thresholds"].values
    y = df.loc[df["model"] == "All", "NB"].values
    lowess = sm.nonparametric.lowess(y, x, frac=frac)
    xnew = lowess[:, 0]
    ynew = lowess[:, 1]
    ax_m.plot(xnew, ynew, "--", label="All")
    ax_c.plot(xnew, ynew, "--", label="All")
    x = df.loc[df["model"].astype(str) == "nan", "thresholds"].values
    y = df.loc[df["model"].astype(str) == "nan", "NB"].values
    ax_m.plot(x, y, "--", label="None")
    ax_c.plot(x, y, "--", label="None")
    groups = df.groupby("model", dropna=False)
    for group in groups:
        x = group[1]["thresholds"].values
        y = group[1]["NB"].values
        lowess = sm.nonparametric.lowess(y, x, frac=frac)
        xnew = lowess[:, 0]
        ynew = lowess[:, 1]
        if str(group[0]) in models:
            ax_m.plot(xnew, ynew, label=group[0])
        elif str(group[0]) in cilncs:
            ax_c.plot(xnew, ynew, label=group[0])
    ax_m.set_xlabel("Risk Threshold", fontdict={'weight': 'bold'})
    ax_m.set_ylabel("Net Benefit", fontdict={'weight': 'bold'})
    ax_m.set_ylim(-0.1, max(df["NB"]))
    ax_m.legend()
    ax_c.set_xlabel("Risk Threshold", fontdict={'weight': 'bold'})
    ax_c.set_ylabel("Net Benefit", fontdict={'weight': 'bold'})
    ax_c.set_ylim(-0.1, max(df["NB"]))
    ax_c.legend()
    return fig_m, fig_c
p_m1, p_c1 = plot_dca(pd_dca_i1, 0.1)
p_mo, p_co = plot_dca(pd_dca_io, 0.1)
p_m1.savefig(
    "/mnt/Projects/Radiomics-CT/99.Manuscript/Plots/DCA.Model.Inst.1.pdf",
    bbox_inches="tight",
)
p_c1.savefig(
    "/mnt/Projects/Radiomics-CT/99.Manuscript/Plots/DCA.Clinic.Inst.1.pdf",
    bbox_inches="tight",
)
p_mo.savefig(
    "/mnt/Projects/Radiomics-CT/99.Manuscript/Plots/DCA.Model.Inst.Other.pdf",
    bbox_inches="tight",
)
p_co.savefig(
    "/mnt/Projects/Radiomics-CT/99.Manuscript/Plots/DCA.Clinic.Inst.Other.pdf",
    bbox_inches="tight",
)
def draw_cal(df:pd.DataFrame):
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot([0,1], [0,1], "--", color="gray")
    ax.plot(df['mean.predicted'], df['KM'],label='Original', color='#2ca02c')
    ax.plot(df['mean.predicted'], df['KM.corrected'],label='Corrected', color='#ff7f0e')
    ax.errorbar(df['mean.predicted'], df['KM'], yerr=df['std.err'], fmt='o', label='Error', color='#1f77b4')
    ax.set_xlabel('Actual 2-years RFS (proportion)', fontdict={'weight': 'bold'})
    ax.set_ylabel('Predicted Probability of 2-years RFS', fontdict={'weight': 'bold'})
    ax.legend()
    return fig
cal_i1 = draw_cal(pd_cal_i1)
cal_io = draw_cal(pd_cal_io)
cal_i1_Tr = draw_cal(pd_cal_i1_Tr)
cal_i1_Te = draw_cal(pd_cal_i1_Te)
cal_i1.savefig("/mnt/Projects/Radiomics-CT/99.Manuscript/Plots/CAL.Inst.1.pdf", bbox_inches='tight')
cal_io.savefig("/mnt/Projects/Radiomics-CT/99.Manuscript/Plots/CAL.Inst.Other.pdf", bbox_inches='tight')
cal_i1_Tr.savefig("/mnt/Projects/Radiomics-CT/99.Manuscript/Plots/CAL.Inst.1.Tr.pdf", bbox_inches='tight')
cal_i1_Te.savefig("/mnt/Projects/Radiomics-CT/99.Manuscript/Plots/CAL.Inst.1.Te.pdf", bbox_inches='tight')