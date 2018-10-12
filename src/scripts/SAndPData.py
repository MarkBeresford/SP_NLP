import pandas as pd
from sklearn import preprocessing


class SAndPData:
    def __init__(self, file_path, feature_columns, company_name_column):
        self.file_path = file_path
        self.df_data = pd.read_csv(self.file_path)
        self.feature_columns = self.df_data[feature_columns]
        self.company_names = self.df_data[company_name_column]

    @staticmethod
    def convert_pandas_df_to_numpy_array(pandas_df):
        return pandas_df.values

    @staticmethod
    def normalise_features(numpy_array):
        return preprocessing.MinMaxScaler().fit_transform(numpy_array)
