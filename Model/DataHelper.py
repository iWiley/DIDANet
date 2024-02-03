import pandas as pd
import numpy as np
import sksurv

from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper

class DataHelper:
    def __init__(self, feature_csv_train, surv_train, stage_csv=None, cut_time=721., col_PatientID=0, col_time=2,
                 col_status=5):
        self.feature_csv_train = feature_csv_train
        self.cut_time = cut_time
        self.dt_train_X = pd.read_csv(feature_csv_train)
        self.dt_train_X = self.dt_train_X.set_index('PatientID')
        f_col = [c for c in self.dt_train_X.columns.values if c != 'PatientID']
        self.x_mapper = DataFrameMapper([([col], StandardScaler()) for col in f_col], df_out=True)
        self.dt_train_X = self.x_mapper.fit_transform(self.dt_train_X).astype('float32')
        dt_s = pd.read_csv(surv_train)
        dt_s = pd.DataFrame({
            'PatientID': dt_s.iloc[:, col_PatientID],
            'time': dt_s.iloc[:, col_time],
            'status': dt_s.iloc[:, col_status].replace(np.nan, 0)
        })
        replace = dt_s['time'] > self.cut_time
        dt_s.loc[replace, 'status'] = 0
        dt_s.loc[replace, 'time'] = self.cut_time
        self.dt_train_Y = dt_s
        self.stage_df = None
        if stage_csv != None:
            stage_df = pd.read_csv(stage_csv)
            self.stage_col = [c for c in stage_df.columns.values if c != 'PatientID']
            stage_mapper = DataFrameMapper([(col, None) for col in self.stage_col])
            stage_df = stage_df[self.stage_col].astype('category').apply(lambda x: x.cat.codes)
            self.stage_df = stage_mapper.fit_transform(stage_df).astype('float32')
    def GetTrain(self):
        dt = pd.merge(self.dt_train_X, self.dt_train_Y, how='left', on='PatientID')
        if self.stage_df != None:
            dt = pd.merge(dt, self.stage_df, how='left', on='PatientID')
        return dt.set_index('PatientID')

    def GetTrain_DL(self):
        dt = self.GetTrain()
        X = dt[[c for c in self.dt_train_X.columns.values if c != 'PatientID']]
        get_data = lambda d: (d['time'].values, d['status'].values)
        Y = get_data(dt)
        if self.stage_df == None:
            return X, Y
        STAGE = dt[self.stage_col]
        return X, Y, STAGE
    def GetTrain_ML(self):
        dt = self.GetTrain()
        X = dt[[c for c in self.dt_train_X.columns.values if c != 'PatientID']]
        if self.stage_col != None:
            STAGE = dt[self.stage_col].astype('category')
        Y = dt[['status', 'time']]
        Y.loc[:, 'status'] = Y.loc[:, 'status'].astype(bool)
        Y = Y.to_records(index=False)
        if self.stage_col == None:
            return X, Y
        return X, Y, STAGE
    def GetTest(self, feature_csv, surv_csv, stage_df=None, col_PatientID=0, col_time=2, col_status=5):
        feature_df = pd.read_csv(feature_csv)
        feature_df = feature_df.rename(columns={feature_df.columns[0]: 'PatientID'})
        feature_df = feature_df.set_index('PatientID')
        feature_df = self.x_mapper.transform(feature_df).astype('float32')
        surv_df = pd.read_csv(surv_csv)
        surv_df = pd.DataFrame({
            'PatientID': surv_df.iloc[:, col_PatientID],
            'time': surv_df.iloc[:, col_time],
            'status': surv_df.iloc[:, col_status].replace(np.nan, 0)
        })
        replace = surv_df['time'] > self.cut_time
        surv_df.loc[replace, 'status'] = 0
        surv_df.loc[replace, 'time'] = self.cut_time
        dt = pd.merge(feature_df, surv_df, how='left', on='PatientID')
        if stage_df != None:
            dt = pd.merge(dt, stage_df, how='left', on='PatientID')
        return dt.set_index('PatientID')
    def GetTest_DL(self, feature_csv, surv_csv, stage_csv=None, col_PatientID=0, col_time=2, col_status=5):
        if stage_csv != None:
            stage_df = pd.read_csv(stage_csv)
            stage_col = [c for c in stage_df.columns.values if c != 'PatientID']
            stage_mapper = DataFrameMapper([(col, None) for col in stage_col])
            stage_df = stage_df[stage_col].astype('category').apply(lambda x: x.cat.codes)
            stage_df = stage_mapper.fit_transform(stage_df).astype('float32')
            dt = self.GetTest(feature_csv, surv_csv, stage_df, col_PatientID, col_time, col_status)
        else:
            dt = self.GetTest(feature_csv, surv_csv, None, col_PatientID, col_time, col_status)
        X = dt[[c for c in self.dt_train_X.columns.values if c != 'PatientID']]
        X = X.to_numpy().astype('float32')
        get_data = lambda d: (d['time'].values, d['status'].values)
        Y = get_data(dt)
        if stage_csv == None:
            return X, Y, dt
        STAGE = dt[stage_col]
        return X, Y, STAGE, dt
    def GetTest_ML(self, feature_csv, surv_csv, stage_csv=None, col_PatientID=0, col_time=2, col_status=5):
        if stage_csv != None:
            stage_df = pd.read_csv(stage_csv)
            stage_col = [c for c in stage_df.columns.values if c != 'PatientID']
            stage_mapper = DataFrameMapper([(col, None) for col in stage_col])
            stage_df = stage_df[stage_col].astype('category').apply(lambda x: x.cat.codes)
            stage_df = stage_mapper.fit_transform(stage_df).astype('float32')
            dt = self.GetTest(feature_csv, surv_csv, stage_df, col_PatientID, col_time, col_status)
        else:
            dt = self.GetTest(feature_csv, surv_csv, None, col_PatientID, col_time, col_status)
        X = dt[[c for c in self.dt_train_X.columns.values if c != 'PatientID']].to_numpy().astype('float32')
        if stage_csv != None:
            STAGE = dt[stage_col].astype('category')
        Y = dt[['status', 'time']]
        Y.loc[:, 'status'] = Y.loc[:, 'status'].astype(bool)
        Y = Y.to_records(index=False)
        if stage_csv == None:
            return X, Y, dt
        return X, Y, STAGE, dt
def to_struct_array(self: (int, int)):
    return sksurv.util.Surv.from_arrays(self[1], self[0])
def LoadData(feature_csv, surv_csv, cut_time=721., stage_csv=None):
    dt_f, dt_s = pd.read_csv(feature_csv), pd.read_csv(surv_csv)
    dt_f_col = [c for c in dt_f.columns.values if c != 'PatientID']
    if stage_csv != None:
        dt_stage = pd.read_csv(stage_csv)
        dt_stage_col = [c for c in dt_stage.columns.values if c != 'PatientID']
    dt_s = pd.DataFrame({
        'PatientID': dt_s.iloc[:, 0],
        'time': dt_s["T"],
        'status': dt_s["E"].replace(np.nan, 0)
    })
    replace = dt_s['time'] > cut_time
    dt_s.loc[replace, 'status'] = 0
    dt_s.loc[replace, 'time'] = cut_time
    dt_f['PatientID'] = dt_f['PatientID'].astype(str)
    dt_s['PatientID'] = dt_s['PatientID'].astype(str)
    dt = pd.merge(dt_f, dt_s, how='left', on='PatientID')
    if stage_csv != None:
        dt = pd.merge(dt, dt_stage, how='left', on='PatientID')
    dt = dt.set_index('PatientID')
    if stage_csv != None:
        return dt, (dt_f_col, dt_stage_col)
    return dt, dt_f_col
def LoadData_DL(feature_csv, surv_csv, cut_time=721., is_test=False, stage_csv=None):
    if stage_csv == None:
        dt, f_col = LoadData(feature_csv, surv_csv, cut_time, stage_csv)
    else:
        dt, (f_col, stage_col) = LoadData(feature_csv, surv_csv, cut_time, stage_csv)
    x_mapper = DataFrameMapper([([col], StandardScaler()) for col in f_col])
    X = x_mapper.fit_transform(dt).astype('float32')
    if stage_csv != None:
        stage_mapper = DataFrameMapper([(col, None) for col in stage_col])
        STAGE = dt[stage_col].astype('category').apply(lambda x: x.cat.codes)
        STAGE = stage_mapper.fit_transform(STAGE).astype('float32')
    get_data = lambda d: (d['time'].values, d['status'].values)
    if is_test:
        Y = get_data(dt)
    else:
        Y = get_data(dt)
    if stage_csv == None:
        return X, Y
    return X, Y, STAGE
def LoadData_ML(feature_csv, surv_csv, cut_time=721., stage_csv=None):
    if stage_csv == None:
        dt, f_col = LoadData(feature_csv, surv_csv, cut_time, stage_csv)
    else:
        dt, (f_col, stage_col) = LoadData(feature_csv, surv_csv, cut_time, stage_csv)

    X = dt[f_col]
    if stage_csv != None:
        STAGE = dt[stage_col].astype('category')
    Y = dt[['status', 'time']]
    Y = Y.copy()
    Y.loc[:, 'status'] = Y.loc[:, 'status'].astype(bool)
    Y = Y.to_records(index=False)

    if stage_csv == None:
        return X, Y
    return X, Y, STAGE