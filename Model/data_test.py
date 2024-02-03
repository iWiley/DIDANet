import numpy as np
import pandas as pd
from DataHelper import LoadData_ML, LoadData_DL

test_Xs, test_Ys = LoadData_ML(feature_csv='/mnt/Projects/Radiomics-CT/99.Manuscript/Data/Inst.Other.X.csv',
                                surv_csv='/mnt/Projects/Radiomics-CT/99.Manuscript/Data/Inst.Other.Y.csv')

test_xs, test_ys = LoadData_DL(feature_csv='/mnt/Projects/Radiomics-CT/99.Manuscript/Data/Inst.Other.X.csv',
                               surv_csv='/mnt/Projects/Radiomics-CT/99.Manuscript/Data/Inst.Other.Y.csv')