import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import gc
import argparse
from time import time
import multiprocessing as mp
import os

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', required=True,
                    help='Path to folder with tsv files')
parser.add_argument('-t', '--transforms', required=True, type=str,
                    help='Enumeration of transforms')
args = parser.parse_args()


class Storage:
    """
        Work with file on storage
    """

    @staticmethod
    def work_on_chunks(df):
        df_with_all_code = []
        feature_column_for_codes = {}
        features = df['features'].str.split(',', n=-1, expand=True)
        features.index = df['id_job']
        columns_name_default = ['features_code']
        feature_names = features[0].unique()

        # TODO: get rid of the constants in the number of features
        for code in feature_names:
            if code not in feature_column_for_codes.keys():
                feature_column = [f'feature_{code}_stand_{i}' for i in
                                  range(256)]
                feature_column_for_codes[code] = feature_column
            else:
                feature_column = feature_column_for_codes[code]

            columns_name = columns_name_default + feature_column
            features.columns = columns_name
            features[feature_column] = features[feature_column].astype('int')
            df_with_all_code.append(features)

        return pd.concat(df_with_all_code)

    def read_tsv(self, file_path,
                 num_processes: int = mp.cpu_count(),
                 chunksize: int = 20000):
        df_chunks = pd.read_csv(file_path, sep='\t',
                                chunksize=chunksize,
                                low_memory=False)

        pool = mp.Pool(num_processes)

        function_list = []
        for df in df_chunks:
            f = pool.apply_async(self.work_on_chunks, [df])
            function_list.append(f)

        result = []
        for f in function_list:
            result.append(f.get(timeout=120)) # timeout in 120 seconds = 2 mins

        df = pd.concat(result)

        feature_names = df['features_code'].unique()
        df.drop('features_code', axis=1, inplace=True)

        return df, feature_names

    @staticmethod
    def write_tsv(path_to_save, df):
        df.to_csv(path_to_save, sep='\t', index=False)


class StandardScore:
    """
        z-score normalization
    """

    # TODO: write test
    def __init__(self):
        self.train_mean = None
        self.train_std = None
        self.test_mean = None

    def fit(self, df, y=None):
        self.train_mean = df.mean()
        self.train_std = df.std()
        return self

    def transform(self, df):
        df = (df - self.train_mean) / self.train_std
        return df

    # TODO: create save and load function for saving fit result from train data
    #   This functions will allow reuse fit data without needs of train data.


class FeatureGeneration:
    """
        Create new features

        Parameters:
        feature_names : list
            The code of the feature
    """

    def __init__(self, **kwargs):
        self.test_mean = None
        self.columns_name = None
        self.row_id_max = None
        self.feature_names = kwargs['feature_names']

    def max_feature_index(self, df, f_name):
        column_name = f'max_feature_{f_name}_index'
        self.row_id_max = df.idxmax(axis=1)
        df[column_name] = self.row_id_max
        df[column_name] = df[column_name].apply(lambda x: x.split('_')[-1])
        df[column_name] = df[column_name].astype('int')

    def max_feature_abs_mean_diff(self, df, f_name):
        dif_column_name = f'max_feature_{f_name}_abs_mean_diff'
        f_column_name = [f'feature_{f_name}_stand_{i}' for i in range(256)]
        df[dif_column_name] = self.test_mean[self.row_id_max].values
        features_row_max = df[f_column_name].max(axis=1)
        df[dif_column_name] = abs(df[dif_column_name] - features_row_max)

    def fit(self, df, y=None):
        self.columns_name = df.columns

        return self

    def transform(self, df):
        self.test_mean = df.mean()
        for f_name in self.feature_names:
            self.max_feature_index(df, f_name)
            self.max_feature_abs_mean_diff(df, f_name)

        return df


class PrePro:
    """
        Create preprocessing pipeline

        Parameters:
        transform_pipe : list
            The code of the feature
    """

    def __init__(self, transform_pipe: list = [],
                 **kwargs):
        self.transform_pipe = transform_pipe
        self.transform_dict = {
                            'standard_score': StandardScore(),
                            'standard_scaler': StandardScaler(),
                            'feature_generation': FeatureGeneration(**kwargs)}
        self.full_pipeline = None
        self.feature_steps = []

        self.init_pipeline()

    def init_pipeline(self):
        if len(self.transform_pipe) == 0:
            raise ValueError(
                f'Neither feature transformation defined')

        if 'standard_score' in self.transform_pipe:
            if 'standard_scaler' in self.transform_pipe:
                raise ValueError(
                    f'Choose one of standard_score or '
                    f'standard_scaler normalization')

        try:
            self.feature_steps.extend([(transform,
                                        self.transform_dict[transform])
                                       for transform in self.transform_pipe])

        except KeyError:
            raise ValueError(
                f'Indicated not defined transformation ')

        self.full_pipeline = Pipeline(self.feature_steps)

    def fit_train(self, df):
        self.full_pipeline.fit(df)

    def transform_test(self, df):
        df = self.full_pipeline.transform(df)

        return df


def run(path: str = "",
        transform_pipe: list = []):
    storage = Storage()
    train_path = os.path.join(path, 'train.tsv')
    df_train, feature_names_train = storage.read_tsv(train_path)

    pipe = PrePro(transform_pipe=transform_pipe,
                  feature_names=feature_names_train)
    pipe.fit_train(df_train)

    # clear RAM
    if df_train.shape[0] > 500000:
        del df_train
        gc.collect()

    test_path = os.path.join(path, 'test.tsv')
    df_test, feature_names_test = storage.read_tsv(test_path)

    df_test = pipe.transform_test(df_test)
    df_test.reset_index(inplace=True)

    test_proc_path = os.path.join(path, 'test_proc.tsv')
    storage.write_tsv(test_proc_path, df=df_test)


if __name__ == '__main__':
    start_time = time()
    run(path=args.path, transform_pipe=args.transforms.split(','))
    print(time() - start_time)
