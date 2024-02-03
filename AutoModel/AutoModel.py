import os
import optuna
import time

import numpy as np
import pandas as pd

from sksurv.svm import (
    FastSurvivalSVM,
    HingeLossSurvivalSVM,
    NaiveSurvivalSVM,
    FastKernelSurvivalSVM,
    MinlipSurvivalAnalysis,
)
from sksurv.linear_model import CoxnetSurvivalAnalysis, IPCRidge
from sksurv.ensemble import (
    RandomSurvivalForest,
    GradientBoostingSurvivalAnalysis,
    ComponentwiseGradientBoostingSurvivalAnalysis,
    ExtraSurvivalTrees,
)
from optuna.trial import Trial
import torchtuples as tt

from . import AutoModel
from . import DataProvider
from AutoModel.ModelType import ModelType
from pycox.models import CoxPH, DeepHitSingle, MTLR
from pycox.evaluation import EvalSurv

import pickle
from tqdm.auto import tqdm
import torch

class AutoModel:
    _version = "0.0.0.1"
    Verbose = False
    def __init__(self, modeltype: ModelType) -> None:
        self.ModelType = modeltype
        self._isTrainned = False
        self._best_cindex = 0.0
        self._best_cindexs = []
        self._best_brier = 1.0
        self._best_briers = []
        self._model = None
        self._best_dl_net_params = None
        self.__current_acc = "0"
        self._best_params = None
        self._best_train_cindex = 0.0
        self._best_train_cindexs = []
        self._best_train_brier = 1.0
        self._best_train_briers = []
        self._lr_finder_result = []
        if torch.cuda.is_available() and ModelType.GetType(self.ModelType) == "DL":
            self.device = torch.device("cuda")
            tqdm.write("GPU will be used for training")
        else:
            self.device = torch.device("cpu")
            tqdm.write("CPU will be used for training")
    def Train(self, dp: DataProvider, hypersearch_times: int, standard: float = 0.05):
        if standard <= 0:
            raise "standard must be greater than 0"

        self._dp = dp
        self._Standarder = dp.Standarder
        self._standard = standard
        self._hypersearch_times = hypersearch_times

        self.EVAL_Y_TIMEPOINTS = np.linspace(
            self._dp.Y.iloc[:, 0].values.min(), self._dp.Y.iloc[:, 0].values.max(), 100
        )
        optuna.logging.disable_default_handler()
        optuna.logging.disable_propagation()
        if hypersearch_times > 0:
            self._study = optuna.create_study(
                direction=optuna.study.StudyDirection.MAXIMIZE,
                sampler=optuna.samplers.TPESampler(seed=dp.RandomSeeds),
            )
            tqdm.write(f"Trainning begin, model type: {self.ModelType}")
            self._bar = tqdm(
                total=hypersearch_times * 100 * self._dp.CrossvaildCount,
                mininterval=1,
            )
            self._study.optimize(
                func=self._objective,
                n_trials=hypersearch_times * 100,
                show_progress_bar=False,
            )
            if self.__sap__ == False:
                self._bar.close()
            self._isTrainned = True
            return (
                self._best_cindex,
                self._best_brier,
                self._best_params,
            )
        else:
            raise "The number of hyperparameter searches cannot be zero or negative"
    def Predict(self, data):
        if not self._isTrainned:
            raise "Please train the model first."
        INDEX, X, Y = self._dp.GetPredictData(data)
        if ModelType.GetType(self.ModelType) == "ML":
            surv_func = self._model.predict_survival_function(X)
            prediction = []
            for patient in surv_func:
                patient_probabilities = []
                for t in self.EVAL_Y_TIMEPOINTS:
                    patient_probabilities.append(patient(t))
                prediction.append(patient_probabilities)
            t = Y[:, 0]
            e = Y[:, 1]
            prediction = pd.DataFrame(prediction, columns=self.EVAL_Y_TIMEPOINTS).T
            ev = EvalSurv(prediction, t, e, censor_surv="km")
            try:
                cindex = ev.concordance_td()
                score = ev.brier_score(np.array(self.EVAL_Y_TIMEPOINTS)).mean()
            except Exception as e:
                cindex = 0.5
                score = 1.0
                if len(e.args) == 1:
                    if e.args[0] == "float division by zero":
                        tqdm.write("WARNING: A division by 0 anomaly has been detected and the C-index has been forced to change to 0.5, the occurrence of which generally proves that you have too few events occurring!")
                    else:
                        raise e
                else:
                    raise e
        else:
            prediction = self._model.predict_surv_df(X)
            ev = EvalSurv(prediction, Y[:, 0], Y[:, 1], censor_surv="km")
            cindex = ev.concordance_td()
            score = ev.brier_score(np.array(self.EVAL_Y_TIMEPOINTS)).mean()
        if isinstance(prediction, pd.DataFrame):
            if (
                self.ModelType == ModelType.DeepHit
                or self.ModelType == ModelType.N_MTLR
            ):
                prediction.index = self.EVAL_Y_TIMEPOINTS
            return (
                cindex,
                score,
                prediction,
            )
        return (
            cindex,
            score,
            pd.DataFrame({"prediction": prediction}, index=self.EVAL_Y_TIMEPOINTS),
        )
    def _objective(self, trial: Trial):
        acc = []
        birer = []
        birer_mean = []
        t_acc = []
        t_birer = []
        t_birer_mean = []
        self._current_taial = trial
        def append_cindex_brier(
            cindex,
            score,
            birer,
            birer_mean,
            acc,
            model,
            t_cindex,
            t_score,
            t_birer,
            t_birer_mean,
            t_acc,
        ):
            acc.append(cindex)
            t_acc.append(t_cindex)
            if len(acc) == self._dp.CrossvaildCount:
                if not isinstance(score, pd.DataFrame) or not isinstance(
                    score, pd.Series
                ):
                    if len(birer) == 0:
                        birer = pd.DataFrame(columns=score.index)
                    birer = birer._append(score)
                    birer_mean.append(score.values.mean())
                else:
                    birer_mean.append(1)

                if not isinstance(t_score, pd.DataFrame) or not isinstance(
                    t_score, pd.Series
                ):
                    if len(t_birer) == 0:
                        t_birer = pd.DataFrame(columns=t_score.index)
                    t_birer = t_birer._append(t_score)
                    t_birer_mean.append(t_score.values.mean())
                else:
                    t_birer_mean.append(1)
                if self._best_cindex < np.mean(acc):
                    mean = np.mean(acc)
                    std_dev = np.std(acc)
                    mean_10_percent = mean * self._standard
                    if (
                        (std_dev <= mean_10_percent and mean < np.mean(t_acc) * .9)
                    ) and (np.mean(t_acc) - mean) < 0.15:
                        self._model = model
                        self._best_params = self._current_taial.params
                        self._best_dl_net_params = net_params
                        self._best_cindex = np.mean(acc)
                        self._best_cindexs = acc
                        self._best_brier = np.mean(birer_mean)
                        self._best_briers = birer

                        self._best_train_cindex = np.mean(t_acc)
                        self._best_train_cindexs = t_acc
                        self._best_train_brier = np.mean(t_birer_mean)
                        self._best_train_briers = t_birer_mean

                        self.__current_acc = f'{"%.3f" % mean}'
                    else:
                        self.__current_acc = f'{"%.3f" % mean}(droped[{"%.3f" % std_dev}/{"%.3f" % mean_10_percent}])'
                    if AutoModel.Verbose:
                        tqdm.write(
                            f"C-Index: ({np.mean(t_acc)}){np.mean(acc)}, brier: ({t_score.values.mean()}){score.values.mean()}, acc: {acc}, brier: {score.values}, current: {self.__current_acc}"
                        )
                else:
                    acc = [0]
            self._bar.set_description(
                f'CI: { "%.3f" % self._best_cindex}, Br: {"%.3f" % self._best_brier}, Current: {self.__current_acc}'
            )
            self._bar.update()
        if ModelType.GetType(self.ModelType) == "ML":
            net_params = None
            modelBuilder, config = AutoModel._buildMLModel(
                self.ModelType, trial, self._dp.RandomSeeds
            )
            dataProvider = self._dp.GetMLData()
            for (x_train, y_train), (x_vaild, y_vaild), y_train_source in dataProvider:
                model = modelBuilder(**config)
                model.fit(x_train, y_train)
                if self.ModelType == ModelType.F_SSVM:
                    prediction = model.predict(x_vaild)
                    t_prediction = model.predict(x_train)
                else:
                    surv_func = model.predict_survival_function(x_vaild)
                    t_surv_func = model.predict_survival_function(x_train)
                    prediction = []
                    t_prediction = []
                    for patient in surv_func:
                        if not np.all(np.array(self.EVAL_Y_TIMEPOINTS) == patient.x):
                            self.EVAL_Y_TIMEPOINTS = patient.x
                        patient_probabilities = []
                        for t in self.EVAL_Y_TIMEPOINTS:
                            patient_probabilities.append(patient(t))
                        prediction.append(patient_probabilities)
                    for patient in t_surv_func:
                        t_patient_probabilities = []
                        for t in self.EVAL_Y_TIMEPOINTS:
                            t_patient_probabilities.append(patient(t))
                        t_prediction.append(t_patient_probabilities)

                t = y_vaild.iloc[:, 0].values
                e = y_vaild.iloc[:, 1].values
                prediction = pd.DataFrame(prediction).T
                ev = EvalSurv(prediction, t, e, censor_surv="km")
                try:
                    cindex = ev.concordance_td()
                    score = ev.brier_score(np.array(self.EVAL_Y_TIMEPOINTS))
                except Exception as e:
                    cindex = 0
                    score = 1
                    if len(e.args) == 1:
                        if e.args[0] == "float division by zero":
                            tqdm.write("WARNING: A division by 0 anomaly has been detected and the C-index has been forced to change to 0.5, the occurrence of which generally proves that you have too few events occurring!")
                        else:
                            raise e
                    else:
                        raise e
                t_t = y_train_source.iloc[:, 0].values
                t_e = y_train_source.iloc[:, 1].values
                t_prediction = pd.DataFrame(t_prediction).T
                t_ev = EvalSurv(t_prediction, t_t, t_e, censor_surv="km")
                try:
                    t_cindex = t_ev.concordance_td()
                    t_score = t_ev.brier_score(np.array(self.EVAL_Y_TIMEPOINTS))
                except Exception as e:
                    t_cindex = 0
                    t_score = 1
                    if len(e.args) == 1:
                        if e.args[0] == "float division by zero":
                            tqdm.write("WARNING: A division by 0 anomaly has been detected and the C-index has been forced to change to 0.5, the occurrence of which generally proves that you have too few events occurring!")
                        else:
                            raise e
                    else:
                        raise e
                append_cindex_brier(
                    cindex,
                    score,
                    birer,
                    birer_mean,
                    acc,
                    model,
                    t_cindex,
                    t_score,
                    t_birer,
                    t_birer_mean,
                    t_acc,
                )
        else:
            num_nodes = [
                trial.suggest_int("layer_1", 5, 30, log=False),
                trial.suggest_int("layer_2", 5, 30, log=False),
            ]
            dropout = trial.suggest_float("dropout", 0.1, 0.9, log=False)
            epochs = trial.suggest_int("epochs", 250, 2000, log=False)
            out_features = (
                1
                if self.ModelType == ModelType.DeepSurv
                else len(self.EVAL_Y_TIMEPOINTS)
            )
            net_params = {
                "num_nodes": num_nodes,
                "dropout": dropout,
                "out_features": out_features,
                "in_features": self._dp.X.shape[1],
                "batch_norm": True,
                "output_bias": False,
            }
            if self.ModelType == ModelType.DeepHit:
                alpha = trial.suggest_float("alpha", 0.1, 0.9, log=False)
                sigma = trial.suggest_float("sigma", 0.1, 0.9, log=False)
            net = tt.practical.MLPVanilla(**net_params)
            callbacks = [
                tt.callbacks.EarlyStopping()
            ]
            dataProvider = self._dp.GetDLData(self.ModelType, self.EVAL_Y_TIMEPOINTS)
            cv_time = 0
            for (
                (x_train, y_train),
                (x_vaild, y_vaild),
                (x_test, y_test),
                (x_trainvai, y_trainvai),
                (y_trainvai_time, y_trainvai_event),
                (y_test_time, y_test_event),
            ) in dataProvider:
                batch_size = len(x_train)
                if self.ModelType == ModelType.DeepSurv:
                    model = CoxPH(net, tt.optim.Adam, device=self.device)
                elif self.ModelType == ModelType.DeepHit:
                    model = DeepHitSingle(
                        net,
                        tt.optim.Adam,
                        alpha=alpha,
                        sigma=sigma,
                        duration_index=self.EVAL_Y_TIMEPOINTS,
                        device=self.device,
                    )
                elif self.ModelType == ModelType.N_MTLR:
                    model = MTLR(
                        net,
                        tt.optim.Adam,
                        duration_index=self.EVAL_Y_TIMEPOINTS,
                        device=self.device,
                    )
                if len(self._lr_finder_result) < self._dp.CrossvaildCount:
                    lrfinder = model.lr_finder(
                        x_train, y_train, batch_size, tolerance=10, shuffle=False
                    )
                    self._lr_finder_result.append(lrfinder.get_best_lr())
                model.optimizer.set_lr(self._lr_finder_result[cv_time])
                cv_time += 1
                model.fit(
                    x_train,
                    y_train,
                    batch_size,
                    epochs,
                    callbacks,
                    verbose=False,
                    val_batch_size=batch_size,
                    val_data=(x_vaild, y_vaild),
                    shuffle=False,
                )
                if self.ModelType == ModelType.DeepSurv:
                    model.compute_baseline_hazards()
                surv = model.predict_surv_df(x_test)
                t_surv = model.predict_surv_df(x_trainvai)
                ev = EvalSurv(surv, y_test_time, y_test_event, censor_surv="km")
                t_ev = EvalSurv(
                    t_surv, y_trainvai_time, y_trainvai_event, censor_surv="km"
                )
                try:
                    cindex = ev.concordance_td()
                    score = ev.brier_score(self.EVAL_Y_TIMEPOINTS)
                except Exception as e:
                    cindex = 0
                    score = 1
                    if len(e.args) == 1:
                        if e.args[0] == "float division by zero":
                            tqdm.write(
                                "WARNING: A division by 0 anomaly has been detected and the C-index has been forced to change to 0.5, the occurrence of which generally proves that you have too few events occurring!"
                            )
                        else:
                            raise e
                    else:
                        raise e
                try:
                    t_cindex = t_ev.concordance_td()
                    t_score = t_ev.brier_score(np.array(self.EVAL_Y_TIMEPOINTS))
                except Exception as e:
                    t_cindex = 0
                    t_score = 1
                    if len(e.args) == 1:
                        if e.args[0] == "float division by zero":
                            tqdm.write("WARNING: A division by 0 anomaly has been detected and the C-index has been forced to change to 0.5, the occurrence of which generally proves that you have too few events occurring!")
                        else:
                            raise e
                    else:
                        raise e
                append_cindex_brier(
                    cindex,
                    score,
                    birer,
                    birer_mean,
                    acc,
                    model,
                    t_cindex,
                    t_score,
                    t_birer,
                    t_birer_mean,
                    t_acc,
                )
        return np.mean(acc)
    def Save(self, file_path):
        if not self._isTrainned:
            raise "Please do the training first."
        if self._model == None:
            raise "No suitable model was found, please adjust the standard parameters to continue the search."
        if ModelType.GetType(self.ModelType) == "DL":
            self._model.save_net("tmp.pt")
            time.sleep(0.5)
            with open("tmp.pt", "rb") as f:
                md = f.read()
            if self.ModelType == ModelType.DeepSurv:
                with open("tmp_blh.pickle", "rb") as f:
                    tmp_blh = f.read()
                os.remove("tmp_blh.pickle")
            os.remove("tmp.pt")
        param = {
            "model": self._model if ModelType.GetType(self.ModelType) == "ML" else md,
            "version": AutoModel._version,
            "modelType": self.ModelType,
            "standarder": self._Standarder,
            "eval_points": self.EVAL_Y_TIMEPOINTS,
            "lr_finder_result": self._lr_finder_result,
            "hypersearch_times": self._hypersearch_times,
            "best_cindex": self._best_cindex,
            "best_cindexs": self._best_cindexs,
            "best_brier": self._best_brier,
            "best_briers": self._best_briers,
            "best_train_cindex": self._best_train_cindex,
            "best_train_cindexs": self._best_train_cindexs,
            "best_train_brier": self._best_train_brier,
            "best_train_briers": self._best_train_briers,
            "best_params": self._best_params,
            "best_dl_net_params": self._best_dl_net_params,
            "DataProvider": self._dp,
            "standard": self._standard,
            "tmp_blh": tmp_blh if self.ModelType == ModelType.DeepSurv else "",
        }
        with open(file_path, "wb") as f:
            pickle.dump(param, f)
    @staticmethod
    def _buildMLModel(modelType: ModelType, trial: Trial, random_state: int):
        if modelType == ModelType.RSF:
            n_jobs = os.cpu_count()
            n_jobs = 1 if n_jobs == None else n_jobs - 1
            config = {
                "n_estimators": trial.suggest_int("n_estimators", 1, 1000),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 100),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 100),
                "n_jobs": n_jobs,
                "random_state": random_state,
            }
            modelBuilder = RandomSurvivalForest
        if modelType == ModelType.ERSF:
            n_jobs = os.cpu_count()
            n_jobs = 1 if n_jobs == None else n_jobs - 1
            config = {
                "n_estimators": trial.suggest_int("n_estimators", 1, 1000),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 100),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 100),
                "min_weight_fraction_leaf": trial.suggest_float(
                    "min_weight_fraction_leaf", 1e-3, 0.5, log=True
                ),
                "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
                "oob_score": trial.suggest_categorical("oob_score", [True, False]),
                "warm_start": trial.suggest_categorical("warm_start", [True, False]),
                "n_jobs": n_jobs,
                "random_state": random_state,
            }
            modelBuilder = ExtraSurvivalTrees
        elif modelType == ModelType.F_SSVM:
            config = {
                "max_iter": trial.suggest_int("max_iter", 100, 1000),
                "tol": trial.suggest_float("tol", 1e-6, 1e-3),
                "random_state": random_state,
            }
            modelBuilder = FastSurvivalSVM
        elif modelType == ModelType.HL_SSVM:
            config = {
                "alpha": trial.suggest_float("alpha", 0.0, 10.0),
                "solver": trial.suggest_categorical("solver", ["ecos", "osqp"]),
                "kernel": trial.suggest_categorical(
                    "kernel",
                    ["linear", "poly", "rbf", "sigmoid", "cosine", "precomputed"],
                ),
                "random_state": random_state,
            }
            modelBuilder = HingeLossSurvivalSVM
        elif modelType == ModelType.N_SSVM:
            config = {
                "alpha": trial.suggest_float("alpha", 0.0, 10.0),
                "loss": trial.suggest_categorical("loss", ["hinge", "squared_hinge"]),
                "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
                "dual": trial.suggest_categorical("dual", [True, False]),
                "tol": trial.suggest_float("tol", 1e-6, 1e-3),
                "random_state": random_state,
            }
            modelBuilder = NaiveSurvivalSVM
        elif modelType == ModelType.FK_SSVM:
            config = {
                "alpha": trial.suggest_float("alpha", 0.0, 10.0),
                "rank_ratio": trial.suggest_float("rank_ratio", 0.0, 1.0),
                "kernel": trial.suggest_categorical(
                    "kernel",
                    ["linear", "poly", "rbf", "sigmoid", "cosine", "precomputed"],
                ),
                "tol": trial.suggest_float("tol", 1e-6, 1e-3),
                "optimizer": trial.suggest_categorical(
                    "optimizer", ["avltree", "rbtree"]
                ),
                "random_state": random_state,
            }
            modelBuilder = FastKernelSurvivalSVM
        elif modelType == ModelType.MINLIP:
            config = {
                "alpha": trial.suggest_float("alpha", 0.0, 10.0),
                "solver": trial.suggest_categorical("solver", ["ecos", "osqp"]),
                "kernel": trial.suggest_categorical(
                    "kernel",
                    ["linear", "poly", "rbf", "sigmoid", "cosine", "precomputed"],
                ),
                "max_iter": trial.suggest_int("max_iter", 100, 1000),
                "pairs": trial.suggest_categorical("pairs", ["all", "nearest", "next"]),
                "random_state": random_state,
            }
            modelBuilder = MinlipSurvivalAnalysis
        elif modelType == ModelType.IPCRidge:
            config = {
                "alpha": trial.suggest_float("alpha", 0.01, 0.5, log=True),
                "random_state": random_state,
            }
            modelBuilder = IPCRidge
        elif modelType == ModelType.CGBSA:
            config = {
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.5, log=True
                ),
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "dropout_rate": trial.suggest_float(
                    "dropout_rate", 0.01, 0.5, log=True
                ),
                "random_state": random_state,
            }
            modelBuilder = ComponentwiseGradientBoostingSurvivalAnalysis
        elif modelType == ModelType.GBSA:
            config = {
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.5, log=True
                ),
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "min_samples_split": trial.suggest_float("min_samples_split", 0.0, 1.0),
                "min_samples_leaf": trial.suggest_float("min_samples_leaf", 0.0, 1.0),
                "min_weight_fraction_leaf": trial.suggest_float(
                    "min_weight_fraction_leaf", 0.0, 0.5
                ),
                "max_depth": trial.suggest_float("max_depth", 1, 5),
                "dropout_rate": trial.suggest_float(
                    "dropout_rate ", 0.01, 0.5, log=True
                ),
            }
            modelBuilder = GradientBoostingSurvivalAnalysis
        elif modelType == ModelType.LASSO:
            config = {
                "l1_ratio": 1,
                "alpha_min_ratio": trial.suggest_float(
                    "alpha_min_ratio", 0.01, 0.5, log=True
                ),
                "normalize": True,
                "fit_baseline_model": True,
            }
            modelBuilder = CoxnetSurvivalAnalysis
        elif modelType == ModelType.Elastic:
            config = {
                "l1_ratio": trial.suggest_float("l1_ratio", 0.01, 0.99, log=True),
                "alpha_min_ratio": trial.suggest_float(
                    "alpha_min_ratio", 0.01, 0.5, log=True
                ),
                "normalize": True,
                "fit_baseline_model": True,
            }
            modelBuilder = CoxnetSurvivalAnalysis

        return modelBuilder, config
    @staticmethod
    def _buildDLModel(modelType: ModelType):
        if modelType == ModelType.DeepSurv:
            return CoxPH
        elif modelType == ModelType.DeepHit:
            return DeepHitSingle
        elif modelType == ModelType.N_MTLR:
            return MTLR
        else:
            raise "Model types that do not exist"
    @staticmethod
    def New(modeltype: ModelType):
        return AutoModel(modeltype)
    @staticmethod
    def Load(am_file: str):
        with open(am_file, "rb") as f:
            param = pickle.load(f)
        model_type = param["modelType"]
        am = AutoModel(model_type)
        version = param["version"]
        am._Standarder = param["standarder"]
        am.EVAL_Y_TIMEPOINTS = param["eval_points"]
        am._lr_finder_result = param["lr_finder_result"]
        am._best_cindex = param["best_cindex"]
        am._best_cindexs = param["best_cindexs"]
        am._best_brier = param["best_brier"]
        am._best_briers = param["best_briers"]
        am._hypersearch_times = param["hypersearch_times"]
        am._best_train_cindex = param["best_train_cindex"]
        am._best_train_cindexs = param["best_train_cindexs"]
        am._best_train_brier = param["best_train_brier"]
        am._best_train_briers = param["best_train_briers"]
        am._best_params = param["best_params"]
        am._dp = param["DataProvider"]
        am._best_dl_net_params = param["best_dl_net_params"]
        am._standard = param["standard"]
        am._isTrainned = True
        if ModelType.GetType(model_type) == "DL":
            if model_type == ModelType.DeepSurv:
                with open("tmp_blh.pickle", "wb") as f:
                    f.write(param["tmp_blh"])
            with open("tmp.pt", "wb") as f:
                f.write(param["model"])
            net = tt.practical.MLPVanilla(**am._best_dl_net_params)
            model = AutoModel._buildDLModel(model_type)
            model = model(net, tt.optim.Adam)
            model.load_net("tmp.pt")
            os.remove("tmp.pt")
            if model_type == ModelType.DeepSurv:
                os.remove("tmp_blh.pickle")
            am._model = model
        else:
            am._model = param["model"]
        return am