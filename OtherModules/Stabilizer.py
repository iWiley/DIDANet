import os
import pandas as pd
import numpy as np
import pickle
import pingouin as pg
from tqdm import tqdm

USE_DROP_SELF = True
CUTOFF_P = 0.05 
CUTOFF_COR = 0.8 
CUT_721 = True

OUT_NAME = f"[{f'DropSelfCor{CUTOFF_COR}' if USE_DROP_SELF else 'nDropSelfCor'}][{f'Cox{CUTOFF_P}' if USE_COX else 'nCox'}][{CUTOFF_COR}][{'CutTime' if CUT_721 else 'nCutTime'}]"

DIR_IN = "/DIR_IN"
DIR_OUT = "/DIR_OUT"

DIR_IN = "/mnt/Projects/Radiomics-CT/DATA"
DIR_OUT = "/mnt/Projects/Radiomics-CT/DATA/NFeature"

feature_a = f"{DIR_IN}/Radiomics-Features.csv"
feature_b = f"{DIR_IN}/Radiomics-Features.Re1.csv"
feature_c = f"{DIR_IN}/Radiomics-Features.Re2.csv"

feature_a = "/mnt/Projects/Radiomics-CT/99.Manuscript/Prediction/X/Train-Val.1.csv"
feature_b = "/mnt/Projects/Radiomics-CT/99.Manuscript/Prediction/X/Train-Val.2.csv"
feature_c = "/mnt/Projects/Radiomics-CT/99.Manuscript/Prediction/X/Train-Val.3.csv"
surv = "/mnt/Projects/Radiomics-CT/99.Manuscript/Prediction/Y/Surv.xlsx"

if os.path.exists(feature_a) == False:
    print("The file Radiomics Features.csv does not exist. Please rename the entire sample file and place it in the input directory!")
    exit()
if os.path.exists(feature_b) == False:
    print("File Radiomics Features Re1.csv does not exist, please rename the first duplicate sample file and place it in the input directory!")
    exit()

if os.path.exists(f'COR.{CUTOFF_COR}.csv') == False:
    data = pd.read_csv(feature_a, index_col=0)
    data_re1 = pd.read_csv(feature_b, index_col=0)
    data_re2 = (
        pd.read_csv(feature_c, index_col=0) if os.path.exists(feature_c) else pd.DataFrame()
    )

    if len(data_re2) == 0:
        print("Warning: File Radiomics Features Re2.csv does not exist, correlation analysis will only be conducted based on the first repeated sample")

    dd = ""
    dd += f"[Org] {data.shape[1]} > "

    data_cor1 = data.loc[data.index.isin(data_re1.index)]
    re_icc1 = []

    for c in tqdm(range(0, len(data_cor1.columns))):
        icc_n = data_cor1.columns[c]

        x = data_cor1[icc_n]
        y = data_re1[icc_n]

        x = x.tolist()
        y = y.tolist()

        df = pd.DataFrame(
            {
                "targets": ["target{}".format(i + 1) for i in range(len(x))] * 2,
                "raters": ["rater1"] * len(x) + ["rater2"] * len(y),
                "ratings": x + y,
            }
        )

        icc = pg.intraclass_corr(
            data=df, targets="targets", raters="raters", ratings="ratings"
        )
        icc1 = icc[icc["Type"] == "ICC1"]["ICC"].values[0]
        re_icc1.append(icc1 > CUTOFF_COR)

    data_out = data.loc[:, re_icc1]

    dd += f"[Re1] {data_out.shape[1]} > "

    print(f"{data_out.shape[1]} > {CUTOFF_COR}")

    if len(data_re2) != 0:
        data_cor2 = data_out.loc[data_out.index.isin(data_re2.index)]
        data_cor2_re = data_re2.loc[data_re2.index.isin(data_cor2.index)]

        re_icc2 = []

        for c in tqdm(range(0, len(data_cor2.columns))):
            icc_n = data_cor2.columns[c]

            x = data_cor2[icc_n]
            y = data_cor2_re[icc_n]

            x = x.tolist()
            y = y.tolist()

            df = pd.DataFrame(
                {
                    "targets": ["target{}".format(i + 1) for i in range(len(x))] * 2,
                    "raters": ["rater1"] * len(x) + ["rater2"] * len(y),
                    "ratings": x + y,
                }
            )

            icc = pg.intraclass_corr(
                data=df, targets="targets", raters="raters", ratings="ratings"
            )
            icc1 = icc[icc["Type"] == "ICC1"]["ICC"].values[0]
            re_icc2.append(icc1 > CUTOFF_COR)

        data_out = data_out.loc[:, re_icc2]
        print(f" {data_out.shape[1]}  > {CUTOFF_COR}")
        dd += f"[Re2] {data_out.shape[1]} > "

    data_out.to_csv(f'COR.{CUTOFF_COR}.csv')
    with open(f"COR.{CUTOFF_COR}.txt", "w") as f:
        f.write(dd)
else:
    with open(f"COR.{CUTOFF_COR}.txt", "r") as f:
        dd = f.read()
    data_out = pd.read_csv(f'COR.{CUTOFF_COR}.csv', index_col=0)

data_out.to_csv("3281_Features.csv")

if USE_DROP_SELF:
    re_c = data_out.corr(method="spearman").abs()
    upper = re_c.where(np.triu(np.ones(re_c.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > CUTOFF_COR)]
    data_out = data_out.drop(to_drop, axis=1)
    print(f" {data_out.shape[1]} ")
    dd += f"[Self] {data_out.shape[1]} > "

data_out.insert(loc=0, column="PatientID", value=data_out.index)
data_out.rename(columns={"Unnamed: 0": "PatientID"}, inplace=True)

data_out.to_csv("383_Features.csv")

data_out.to_csv(f"{DIR_OUT}/{OUT_NAME}-Features.csv", index=False)
with open(f"{DIR_OUT}/{OUT_NAME}-DownDim.csv", "w") as f:
    f.write(dd)