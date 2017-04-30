import os
import sys

import numpy as np
import pandas as pd

from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler

file_path = os.path.dirname(os.path.realpath(__file__))
lib_path = os.path.abspath(os.path.join(file_path, '..', 'utils'))
sys.path.append(lib_path)

from data_utils import get_file


P1B3_URL = 'http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/P1B3/'


def load_dose_response(min_logconc=-5., max_logconc=-5., subsample=None):
    """Load cell line response to different drug compounds, sub-select response for a specific
        drug log concentration range and return a pandas dataframe.

    Parameters
    ----------
    path: string
        path to 'NCI60_dose_response_with_missing_z5_avg.csv'
    min_logconc : -3, -4, -5, -6, -7, optional (default -5)
        min log concentration of drug to return cell line growth
    max_logconc : -3, -4, -5, -6, -7, optional (default -5)
        max log concentration of drug to return cell line growth
    subsample: None, 'naive_balancing' (default None)
        subsampling strategy to use to balance the data based on growth
    """

    path = get_file(P1B3_URL + 'NCI60_dose_response_with_missing_z5_avg.csv')

    df = pd.read_csv(path, sep=',', engine='c',
                     na_values=['na','-',''],
                     dtype={'NSC':object, 'CELLNAME':str, 'LOG_CONCENTRATION':np.float32, 'GROWTH':np.float32})

    df = df[(df['LOG_CONCENTRATION'] >= min_logconc) & (df['LOG_CONCENTRATION'] <= max_logconc)]

    df = df[['NSC', 'CELLNAME', 'GROWTH', 'LOG_CONCENTRATION']]

    if subsample and subsample == 'naive_balancing':
        df1 = df[df['GROWTH'] <= 0]
        df2 = df[(df['GROWTH'] > 0) & (df['GROWTH'] < 50)].sample(frac=0.7, random_state=SEED)
        df3 = df[(df['GROWTH'] >= 50) & (df['GROWTH'] <= 100)].sample(frac=0.18, random_state=SEED)
        df4 = df[df['GROWTH'] > 100].sample(frac=0.01, random_state=SEED)
        df = pd.concat([df1, df2, df3, df4])

    df = df.set_index(['NSC'])

    return df


def load_drug_autoencoded_AG(ncols=None, scaling='std'):
    """Load drug latent representation from Aspuru-Guzik's variational
    autoencoder, sub-select columns of drugs randomly if specificed,
    impute and scale the selected data, and return a pandas dataframe.

    Parameters
    ----------
    path: string
        path to 'Aspuru-Guzik_NSC_latent_representation_292D.csv'
    ncols : int or None
        number of columns (drug latent representations) to randomly subselect (default None : use all data)
    scaling : 'maxabs' [-1,1], 'minmax' [0,1], 'std', or None, optional (default 'std')
        type of scaling to apply

    """
    path = get_file(P1B3_URL + 'Aspuru-Guzik_NSC_latent_representation_292D.csv')

    df = pd.read_csv(path, engine='c', dtype=np.float32)

    df1 = pd.DataFrame(df.loc[:, 'NSC'].astype(int).astype(str))
    df2 = df.drop('NSC', 1)

    total = df2.shape[1]
    if ncols and ncols < total:
        usecols = np.random.choice(total, size=ncols, replace=False)
        df2 = df2.iloc[:, usecols]

    df2 = impute_and_scale(df2, scaling)
    df2 = df2.astype(np.float32)

    df = pd.concat([df1, df2], axis=1)

    return df


def impute_and_scale(df, scaling='std'):
    """Impute missing values with mean and scale data included in pandas dataframe.

    Parameters
    ----------
    df : pandas dataframe
        dataframe to impute and scale
    scaling : 'maxabs' [-1,1], 'minmax' [0,1], 'std', or None, optional (default 'std')
        type of scaling to apply
    """

    df = df.dropna(axis=1, how='all')

    imputer = Imputer(strategy='mean', axis=0)
    mat = imputer.fit_transform(df)

    if scaling is None or scaling.lower() == 'none':
        return pd.DataFrame(mat, columns=df.columns)

    if scaling == 'maxabs':
        scaler = MaxAbsScaler()
    elif scaling == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()

    mat = scaler.fit_transform(mat)

    df = pd.DataFrame(mat, columns=df.columns)

    return df
