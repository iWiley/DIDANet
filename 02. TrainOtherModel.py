#### Training of other models
# The purpose of this file is to train other models to be used as a comparison to our model. 
# > Note that this file supports GPU training for deep learning related models, so if you want to use it to speed up the training process, make sure you have installed the CUDA related drivers and run this file with a CUDA-enabled Nvidia graphics card.
import pandas as pd
import numpy as np
from AutoModel import *
# We have encapsulated a helper class AutoModel to help us train as well as predict other models, the source code of which is provided along with the manuscript.
# For AutoModel related source code, please refer to the AutoModel folder in the same directory.
file_x_path = "/mnt/Projects/Radiomics-CT/99.Manuscript/Data/Inst.1.X.csv"
file_y_path = "/mnt/Projects/Radiomics-CT/99.Manuscript/Data/Inst.1.Y.csv"
model_save_path = "/mnt/Projects/Radiomics-CT/99.Manuscript/Model"
def train(modelType):
    x = pd.read_csv(file_x_path, header=0, index_col=0, low_memory=False)
    y = pd.read_csv(file_y_path, header=0, index_col=0)
    x["T"] = y["T"].values
    x["E"] = y["E"].values
    
    x.reset_index(drop=True, inplace=True)
    dp = DataProvider(
            data=x,
            randomSeeds=666666,
            spilt_ml=(148, 40),
            spilt_dl=(118, 30, 40),
            is_stardandize=True,
            is_shuffle=True,
            crossvaild_count=1,
            col_name_index=None,
            col_name_time="T",
            col_name_event="E",
            col_name_ignore=[],
        )
    model = AutoModel.New(modelType)
    best_cindex, best_brier, _ = model.Train(
        dp, hypersearch_times=5, standard = .05
    )
    model.Save(f"{model_save_path}/{str(modelType).removeprefix('ModelType.')}.am")
    print(
        f"best cindex: {best_cindex}, brier: {best_brier}, cindexs: {model._best_cindexs}"
    )
    print(
        f"best train_cindex: {model._best_train_cindex}, train_brier: {model._best_train_brier}, train_cindexs: {model._best_train_cindexs}"
    )
# train(ModelType.DeepSurv)
# train(ModelType.DeepHit)
# train(ModelType.N_MTLR)
# train(ModelType.RSF)
train(ModelType.LASSO)
# train(ModelType.Elastic)