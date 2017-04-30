import os
import sys

import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

file_path = os.path.dirname(os.path.realpath(__file__))
lib_path = os.path.abspath(file_path)
sys.path.append(lib_path)

from datasets import nci60


def main():
    df = nci60.load_dose_response()
    cell = 'MCF7'
    df2 = df[df['CELLNAME'] == 'BR:MCF7'].reset_index()[['NSC', 'GROWTH', 'LOG_CONCENTRATION']]
    df3 = nci60.load_drug_autoencoded_AG()
    df4 = pd.merge(df2, df3, on='NSC').set_index('NSC')

    data = df4.as_matrix()
    x = data[:, 1:]
    y = data[:, 0]

    model = RandomForestRegressor()
    scores = cross_val_score(model, x, y, cv=2)


if __name__ == '__main__':
    main()
