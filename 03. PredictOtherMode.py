#### Model prediction
# The file is used to predict risk indices for each patient using other models. `02. TrainOtherModel.py` in the same directory as this file is available for model training.
import os
import pandas as pd
from AutoModel import *
file_x1_path = "/mnt/Projects/Radiomics-CT/99.Manuscript/Data/Inst.1.X.csv"
file_y1_path = "/mnt/Projects/Radiomics-CT/99.Manuscript/Data/Inst.1.Y.csv"
file_xo_path = "/mnt/Projects/Radiomics-CT/99.Manuscript/Data/Inst.Other.X.csv"
file_yo_path = "/mnt/Projects/Radiomics-CT/99.Manuscript/Data/Inst.Other.Y.csv"
model_save_path = "/mnt/Projects/Radiomics-CT/99.Manuscript/Model"
x1 = pd.read_csv(file_x1_path, header=0, index_col=0, low_memory=False)
y1 = pd.read_csv(file_y1_path, header=0, index_col=0, low_memory=False)
xo = pd.read_csv(file_xo_path, header=0, index_col=0, low_memory=False)
yo = pd.read_csv(file_yo_path, header=0, index_col=0, low_memory=False)
x1["T"] = y1["T"].values
x1["E"] = y1["E"].values
xo["T"] = yo["T"].values
xo["E"] = yo["E"].values
x1.reset_index(drop=True, inplace=True)
xo.reset_index(drop=True, inplace=True)
for m in os.listdir(model_save_path):
    model = AutoModel.Load(f"{model_save_path}/{m}")
    m = m.removesuffix('.am')
    cindex, brier, risk = model.Predict(x1)
    if not os.path.exists(f"/mnt/Projects/Radiomics-CT/99.Manuscript/Result/{m}"):
        os.mkdir(f"/mnt/Projects/Radiomics-CT/99.Manuscript/Result/{m}")
    risk.to_csv(
        f"/mnt/Projects/Radiomics-CT/99.Manuscript/Result/{m}/Inst.1.csv"
    )
    print(f"{m}: Inst1 cindex: {cindex} brier: {brier}")
    cindex, brier, risk = model.Predict(xo)
    risk.to_csv(
        f"/mnt/Projects/Radiomics-CT/99.Manuscript/Result/{m}/Inst.Other.csv"
    )
    print(f"{m}: OtherInst cindex: {cindex} brier: {brier}")