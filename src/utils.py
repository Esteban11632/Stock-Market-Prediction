import pandas as pd

def get_column_normalized_to_1d(df, column_name):
    column = df[column_name].squeeze()
    if isinstance(column, pd.DataFrame):
        column = column.iloc[:, 0]
    return column