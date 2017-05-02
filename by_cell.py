from __future__ import print_function

import argparse
import os

from datasets import nci60
from regression import regress
from argparser import get_parser


def test1():
    df = nci60.load_by_cell_data()
    regress('XGBoost', df)


def test2():
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=20)
    df = nci60.load_by_cell_data()
    regress(model, df, cv=2)


def main():
    parser = get_parser()
    args = parser.parse_args()

    cells = nci60.all_cells() if 'all' in args.cells else args.cells

    for cell in cells:
        print('-' * 10, 'Cell line:', cell, '-' * 10)
        df = nci60.load_by_cell_data(cell, drug_features=args.drug_features, scaling=args.scaling,
                                     min_logconc=args.min_logconc, max_logconc=args.max_logconc,
                                     subsample=args.subsample, feature_subsample=args.feature_subsample)
        if not df.shape[0]:
            print('No response data found')
            continue
        for model in args.models:
            regress(model, df, cv=args.cv, threads=args.threads, prefix=os.path.join(args.out_dir, cell))


if __name__ == '__main__':
    main()
