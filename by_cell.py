from __future__ import print_function

import argparse

from datasets import nci60
from regression import regress


def test1():
    x, y = nci60.load_by_cell_data()
    regress('XGBoost', x, y)


def test2():
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=20)
    x, y = nci60.load_by_cell_data()
    regress(model, x, y, cv=2)


def get_parser():
    parser = argparse.ArgumentParser(prog='p1b3_baseline')
    parser.add_argument("-c", "--cells", nargs='+', default=['BR:MCF7'],
                        help="list of cell line names")
    parser.add_argument("-d", "--drug_features", nargs='+', default=['descriptors'],
                        choices=['descriptors', 'latent', 'all', 'noise'],
                        help="use dragon7 descriptors, latent representations from Aspuru-Guzik's SMILES autoencoder, or both, or random features; 'descriptors','latent', 'all', 'noise'")
    parser.add_argument("-m", "--models", nargs='+', default=['XGBoost', 'Linear'],
                        help="list of regression models")
    parser.add_argument("--cv", type=int, default=5,
                        help="cross validation folds")
    parser.add_argument("--feature_subsample", type=int, default=0,
                        help="number of features to randomly sample from each category, 0 means using all features")
    parser.add_argument("--min_logconc", type=float, default=-5.0,
                        help="min log concentration of dose response data to use: -3.0 to -7.0")
    parser.add_argument("--max_logconc",  type=float, default=-5.0,
                        help="max log concentration of dose response data to use: -3.0 to -7.0")
    parser.add_argument("--scaling", default='std',
                        choices=['minabs', 'minmax', 'std', 'none'],
                        help="type of feature scaling; 'minabs': to [-1,1]; 'minmax': to [0,1], 'std': standard unit normalization; 'none': no normalization")
    parser.add_argument("--subsample", default='naive_balancing',
                        choices=['naive_balancing', 'none'],
                        help="dose response subsample strategy; 'none' or 'naive_balancing'")
    parser.add_argument("--threads", default=-1,
                        help="number of threads per machine learning training job; -1 for using all threads")
    parser.add_argument("--out", default='out',
                        help="output prefix")
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    cells = nci60.all_cells() if 'all' in args.cells else args.cells

    for cell in cells:
        print('-' * 10, 'Cell line:', cell, '-' * 10)
        x, y = nci60.load_by_cell_data(cell, drug_features=args.drug_features, scaling=args.scaling,
                                       min_logconc=args.min_logconc, max_logconc=args.max_logconc,
                                       subsample=args.subsample, feature_subsample=args.feature_subsample)
        for model in args.models:
            regress(model, x, y, cv=args.cv, threads=args.threads)


if __name__ == '__main__':
    main()
