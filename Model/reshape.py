import numpy as np
import pandas as pd

def reshape(surv):
    x_values = [sf.x for sf in surv]
    y_values = [sf.y for sf in surv]
    print("-------------------------")
    rows = len(x_values[0]) 
    cols = surv.shape[0] 
    data = np.empty((rows, cols))
    df_RSF = pd.DataFrame(data)
    df_RSF.columns = [f'{i}' for i in range(0, cols)]
    x = []
    for i in range(len(x_values[0])):
        x.append(x_values[0][i])
    df_RSF.index = x
    df_RSF = df_RSF.rename_axis('duration')
    for column_name, y_value in zip(df_RSF.columns, y_values):
        df_RSF[column_name] = y_value
    return df_RSF