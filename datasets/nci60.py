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


global_cache = {}

P1B3_URL = 'http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/P1B3/'


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


def load_dose_response(min_logconc=-5., max_logconc=-5., subsample=None, fraction=False):
    """Load cell line response to different drug compounds, sub-select response for a specific
        drug log concentration range and return a pandas dataframe.

    Parameters
    ----------
    min_logconc : -3, -4, -5, -6, -7, optional (default -5)
        min log concentration of drug to return cell line growth
    max_logconc : -3, -4, -5, -6, -7, optional (default -5)
        max log concentration of drug to return cell line growth
    subsample: None, 'naive_balancing' (default None)
        subsampling strategy to use to balance the data based on growth
    fraction: bool (default False)
        divide growth percentage by 100
    """

    path = get_file(P1B3_URL + 'NCI60_dose_response_with_missing_z5_avg.csv')

    df = global_cache.get(path)
    if df is None:
        df = pd.read_csv(path, sep=',', engine='c',
                         na_values=['na', '-', ''],
                         dtype={'NSC':object, 'CELLNAME':str, 'LOG_CONCENTRATION':np.float32, 'GROWTH':np.float32})
        global_cache[path] = df

    df = df[(df['LOG_CONCENTRATION'] >= min_logconc) & (df['LOG_CONCENTRATION'] <= max_logconc)]

    df = df[['NSC', 'CELLNAME', 'GROWTH', 'LOG_CONCENTRATION']]

    if subsample and subsample == 'naive_balancing':
        df1 = df[df['GROWTH'] <= 0]
        df2 = df[(df['GROWTH'] > 0) & (df['GROWTH'] < 50)].sample(frac=0.7, random_state=SEED)
        df3 = df[(df['GROWTH'] >= 50) & (df['GROWTH'] <= 100)].sample(frac=0.18, random_state=SEED)
        df4 = df[df['GROWTH'] > 100].sample(frac=0.01, random_state=SEED)
        df = pd.concat([df1, df2, df3, df4])

    if fraction:
        df['GROWTH'] /= 100

    df = df.set_index(['NSC'])

    return df


def load_drug_descriptors(ncols=None, scaling='std'):
    """Load drug descriptor data, sub-select columns of drugs descriptors
        randomly if specificed, impute and scale the selected data, and return a
        pandas dataframe.

    Parameters
    ----------
    ncols : int or None
        number of columns (drugs descriptors) to randomly subselect (default None : use all data)
    scaling : 'maxabs' [-1,1], 'minmax' [0,1], 'std', or None, optional (default 'std')
        type of scaling to apply
    """

    path = get_file(P1B3_URL + 'descriptors.2D-NSC.5dose.filtered.txt')

    df = global_cache.get(path)
    if df is None:
        df = pd.read_csv(path, sep='\t', engine='c',
                         na_values=['na','-',''],
                         dtype=np.float32)
        global_cache[path] = df

    df1 = pd.DataFrame(df.loc[:,'NAME'].astype(int).astype(str))
    df1.rename(columns={'NAME': 'NSC'}, inplace=True)

    df2 = df.drop('NAME', 1)

    # # Filter columns if requested

    total = df2.shape[1]
    if ncols and ncols < total:
        usecols = np.random.choice(total, size=ncols, replace=False)
        df2 = df2.iloc[:,usecols]

    df2 = impute_and_scale(df2, scaling)
    df2 = df2.astype(np.float32)

    df_dg = pd.concat([df1, df2], axis=1)

    return df_dg


def load_drug_autoencoded_AG(ncols=None, scaling='std'):
    """Load drug latent representation from Aspuru-Guzik's variational
    autoencoder, sub-select columns of drugs randomly if specificed,
    impute and scale the selected data, and return a pandas dataframe

    Parameters
    ----------
    ncols : int or None
        number of columns (drug latent representations) to randomly subselect (default None : use all data)
    scaling : 'maxabs' [-1,1], 'minmax' [0,1], 'std', or None, optional (default 'std')
        type of scaling to apply

    """
    path = get_file(P1B3_URL + 'Aspuru-Guzik_NSC_latent_representation_292D.csv')

    df = global_cache.get(path)
    if df is None:
        df = pd.read_csv(path, engine='c', dtype=np.float32)
        global_cache[path] = df

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


def all_cells():
    df = load_dose_response()
    return df['CELLNAME'].drop_duplicates().tolist()


def all_drugs():
    df = load_dose_response()
    return df['NSC'].drop_duplicates().tolist()


def load_by_cell_data(cell='BR:MCF7', drug_features=['descriptors'], shuffle=True,
                      min_logconc=-5., max_logconc=-4., subsample='naive_balancing',
                      feature_subsample=None, scaling='std', scramble=False):

    """Load dataframe for by cellline models

    Parameters
    ----------
    drug_features: list of strings from 'descriptors', 'latent', 'all', 'noise' (default ['descriptors'])
        use dragon7 descriptors, latent representations from Aspuru-Guzik's SMILES autoencoder
        trained on NSC drugs, or both; use random features if set to noise

    """

    if 'all' in drug_features:
        drug_features = ['descriptors', 'latent']

    df_resp = load_dose_response(fraction=True)

    df = df_resp[df_resp['CELLNAME'] == cell].reset_index()
    df = df[['NSC', 'GROWTH', 'LOG_CONCENTRATION']]
    df = df.rename(columns={'LOG_CONCENTRATION': 'LCONC'})
    print(df.shape)

    for fea in drug_features:
        if fea == 'descriptors':
            df_desc = load_drug_descriptors(ncols=feature_subsample, scaling=scaling)
            df = df.merge(df_desc, on='NSC')
        elif fea == 'latent':
            df_ag = load_drug_autoencoded_AG(ncols=feature_subsample, scaling=scaling)
            df = df.merge(df_ag, on='NSC')
        elif fea == 'noise':
            df_drug_ids = df[['NSC']].drop_duplicates()
            noise = np.random.normal(size=(df_drug_ids.shape[0], 500))
            df_rand = pd.DataFrame(noise, index=df_drug_ids['NSC'],
                                   columns=['RAND-{:03d}'.format(x) for x in range(500)])
            df = df.merge(df_rand, on='NSC')

    df = df.set_index('NSC')

    return df
