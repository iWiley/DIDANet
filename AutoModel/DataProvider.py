import random
from typing import Literal
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
from . import ModelType
from pycox.preprocessing.label_transforms import LabTransDiscreteTime
import sksurv

class DataProvider:
    def __init__(
        self,
        data: pd.DataFrame,
        randomSeeds: int,
        spilt_ml: tuple[int, int],
        spilt_dl: tuple[int, int, int],
        is_stardandize: bool,
        is_shuffle: bool,
        crossvaild_count: int,
        col_name_index: str | None,
        col_name_time: str,
        col_name_event: str,
        col_name_ignore: list[str],
        array_ml: tuple[list[int], list[int]] = None,
        array_dl: tuple[list[int], list[int]] = None,
    ) -> None:
        self.RandomSeeds = randomSeeds
        torch.manual_seed(self.RandomSeeds)
        torch.cuda.manual_seed_all(self.RandomSeeds)
        np.random.seed(self.RandomSeeds)
        random.seed(self.RandomSeeds)
        self.IsStandardize = is_stardandize
        self.TimeColName = col_name_time
        self.EventColName = col_name_event
        self.IndexColName = col_name_index
        self.IgroneCols = col_name_ignore
        self.CrossvaildCount = crossvaild_count
        self.Data = data
        self._dta_dl = []
        self._dta_ml = []
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        if is_shuffle:
            self.Data = self.Data.sample(frac=1)
        self.Index, self.X, self.Y = DataProvider._processData(
            self.Data,
            self.IndexColName,
            self.TimeColName,
            self.EventColName,
            self.IgroneCols,
        )
        if self.IsStandardize:
            self.Standarder = DataFrameMapper(
                [([col], StandardScaler()) for col in self.X.columns.values],
                df_out=True,
            )
            self.X = self.Standarder.fit_transform(self.X).astype("float32")
        self._dl_idx = [
            (
                list(range(spilt_dl[0])),
                list(range(spilt_dl[0], spilt_dl[0] + spilt_dl[1])),
                list(
                    range(
                        spilt_dl[0] + spilt_dl[1],
                        spilt_dl[0] + spilt_dl[1] + spilt_dl[2],
                    )
                ),
            )
        ]
        self._ml_idx = [
            (
                list(range(spilt_ml[0])),
                list(range(spilt_ml[0], spilt_ml[0] + spilt_ml[1])),
            )
        ]
        indexs = list(range(len(self.X)))
        self._dl_idx = []
        self._ml_idx = []
        for i in range(self.CrossvaildCount):
            random.shuffle(indexs)
            dl_idx = (
                indexs[: spilt_dl[0]],
                indexs[spilt_dl[0] : spilt_dl[0] + spilt_dl[1]],
                indexs[spilt_dl[0] + spilt_dl[1] :],
            )
            ml_idx = (indexs[: spilt_ml[0]], indexs[spilt_ml[0] :])
            self._dl_idx.append(dl_idx)
            self._ml_idx.append(ml_idx)
    @staticmethod
    def _processData(
        data: pd.DataFrame,
        col_name_index: str,
        col_name_time: str,
        col_name_event: str,
        col_name_ignore: list[str],
    ):
        if col_name_index != None:
            index = data[col_name_index]
        else:
            index = data.index
        x = data[
            [
                i
                for i in data.columns.values
                if i not in col_name_ignore
                and i != col_name_index
                and i != col_name_time
                and i != col_name_event
            ]
        ].astype("float32")
        y = data[[col_name_time, col_name_event]]
        if y[col_name_time].hasnans:
            raise "There can be no null values in the Time of Y"
        if y[col_name_event].hasnans:
            y[col_name_event] = y[col_name_event].fillna(0).copy()
            print("WARNING: A null value has been detected in the Event of Y, which has been replaced with 0. Please check the data!")
        return index, x, y
    def GetData(self, dataType: Literal["ML", "DL"]):
        if dataType == "DL":
            return self.GetDLData()
        elif dataType == "ML":
            return self.GetMLData()
        else:
            raise "Model type error"
    def GetMLData(self):
        if len(self._dta_ml) == 0:
            for i in self._ml_idx:
                y_t = sksurv.util.Surv.from_arrays(
                    self.Y.iloc[i[0]].iloc[:, 1].astype(bool),
                    self.Y.iloc[i[0]].iloc[:, 0],
                )
                self._dta_ml.append(
                    (
                        (self.X.iloc[i[0]], y_t),
                        (self.X.iloc[i[1]], self.Y.iloc[i[1]]),
                        self.Y.iloc[i[0]],
                    )
                )
        return self._dta_ml
    def GetDLData(self, modelType: ModelType, train_Y_TimePoints: list[float]):
        if len(self._dta_dl) == 0:
            for i in self._dl_idx:
                if modelType == ModelType.DeepHit or modelType == ModelType.N_MTLR:
                    labtrans = LabTransDiscreteTime(train_Y_TimePoints)
                    train_y = labtrans.transform(
                        *(
                            self.Y.iloc[i[0]][self.TimeColName].values,
                            self.Y.iloc[i[0]][self.EventColName].values,
                        )
                    )
                    vaild_y = labtrans.transform(
                        *(
                            self.Y.iloc[i[1]][self.TimeColName].values,
                            self.Y.iloc[i[1]][self.EventColName].values,
                        )
                    )
                    trainvai_y = labtrans.transform(
                        *(
                            np.append(
                                self.Y.iloc[i[0]][self.TimeColName].values,
                                self.Y.iloc[i[1]][self.TimeColName].values,
                            ),
                            np.append(
                                self.Y.iloc[i[0]][self.EventColName].values,
                                self.Y.iloc[i[1]][self.EventColName].values,
                            ),
                        )
                    )
                    train_y = list(train_y)
                    train_y[0] = torch.from_numpy(train_y[0]).to(self.device)
                    train_y[1] = torch.from_numpy(train_y[1]).to(self.device)
                    train_y = tuple(train_y)
                    vaild_y = list(vaild_y)
                    vaild_y[0] = torch.from_numpy(vaild_y[0]).to(self.device)
                    vaild_y[1] = torch.from_numpy(vaild_y[1]).to(self.device)
                    vaild_y = tuple(vaild_y)
                    trainvai_y = list(trainvai_y)
                    trainvai_y[0] = torch.from_numpy(trainvai_y[0]).to(self.device)
                    trainvai_y[1] = torch.from_numpy(trainvai_y[1]).to(self.device)
                    trainvai_y = tuple(trainvai_y)
                else:
                    train_y = (
                        torch.from_numpy(self.Y.iloc[i[0]][self.TimeColName].values).to(
                            self.device
                        ),
                        torch.from_numpy(
                            self.Y.iloc[i[0]][self.EventColName].values
                        ).to(self.device),
                    )
                    vaild_y = (
                        torch.from_numpy(self.Y.iloc[i[1]][self.TimeColName].values).to(
                            self.device
                        ),
                        torch.from_numpy(
                            self.Y.iloc[i[1]][self.EventColName].values
                        ).to(self.device),
                    )
                    trainvai_y = (
                        torch.from_numpy(
                            np.append(
                                self.Y.iloc[i[0]][self.TimeColName].values,
                                self.Y.iloc[i[1]][self.TimeColName].values,
                            )
                        ).to(self.device),
                        torch.from_numpy(
                            np.append(
                                self.Y.iloc[i[0]][self.EventColName].values,
                                self.Y.iloc[i[1]][self.EventColName].values,
                            )
                        ).to(self.device),
                    )

                test_y = (
                    torch.from_numpy(self.Y.iloc[i[2]][self.TimeColName].values).to(
                        self.device
                    ),
                    torch.from_numpy(self.Y.iloc[i[2]][self.EventColName].values).to(
                        self.device
                    ),
                )
                self._dta_dl.append(
                    (
                        (
                            torch.from_numpy(self.X.iloc[i[0]].values).to(self.device),
                            train_y,
                        ),
                        (
                            torch.from_numpy(self.X.iloc[i[1]].values).to(self.device),
                            vaild_y,
                        ),
                        (
                            torch.from_numpy(self.X.iloc[i[2]].values).to(self.device),
                            test_y,
                        ),
                        (
                            torch.from_numpy(
                                np.concatenate(
                                    (
                                        self.X.iloc[i[0]].values,
                                        self.X.iloc[i[1]].values,
                                    ),
                                    axis=0,
                                )
                            ).to(self.device),
                            trainvai_y,
                        ),
                        (
                            np.append(
                                self.Y.iloc[i[0]][self.TimeColName].values,
                                self.Y.iloc[i[1]][self.TimeColName].values,
                            ),
                            np.append(
                                self.Y.iloc[i[0]][self.EventColName].values,
                                self.Y.iloc[i[1]][self.EventColName].values,
                            ),
                        ),
                        (
                            self.Y.iloc[i[2]][self.TimeColName].values,
                            self.Y.iloc[i[2]][self.EventColName].values,
                        ),
                    )
                )
        return self._dta_dl

    def GetPredictData(self, data):
        index, x, y = DataProvider._processData(
            data,
            self.IndexColName,
            self.TimeColName,
            self.EventColName,
            self.IgroneCols,
        )
        if self.IsStandardize:
            x = self.Standarder.transform(x)
        return index, x.to_numpy(), y.to_numpy()