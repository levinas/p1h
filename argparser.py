import argparse


CELLS = ['BR:MCF7']
DRUG_FEATURES = ['descriptors']
MODELS = ['XGBoost', 'Ridge', 'RandomForest']
CV = 5
FEATURE_SUBSAMPLE = 0
MIN_LOGCONC = -5.0
MAX_LOGCONC = -4.0
SCALING = 'std'
SUBSAMPLE = None
THREADS = -1
OUT_DIR = '.'


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cells", nargs='+', default=CELLS,
                        help="list of cell line names")
    parser.add_argument("-d", "--drug_features", nargs='+', default=DRUG_FEATURES,
                        choices=['descriptors', 'latent', 'all', 'noise'],
                        help="use dragon7 descriptors, latent representations from Aspuru-Guzik's SMILES autoencoder, or both, or random features; 'descriptors','latent', 'all', 'noise'")
    parser.add_argument("-m", "--models", nargs='+', default=MODELS,
                        help="list of regression models")
    parser.add_argument("--cv", type=int, default=CV,
                        help="cross validation folds")
    parser.add_argument("--feature_subsample", type=int, default=FEATURE_SUBSAMPLE,
                        help="number of features to randomly sample from each category, 0 means using all features")
    parser.add_argument("--min_logconc", type=float, default=MIN_LOGCONC,
                        help="min log concentration of dose response data to use: -3.0 to -7.0")
    parser.add_argument("--max_logconc",  type=float, default=MAX_LOGCONC,
                        help="max log concentration of dose response data to use: -3.0 to -7.0")
    parser.add_argument("--scaling", default=SCALING,
                        choices=['minabs', 'minmax', 'std', 'none'],
                        help="type of feature scaling; 'minabs': to [-1,1]; 'minmax': to [0,1], 'std': standard unit normalization; 'none': no normalization")
    parser.add_argument("--subsample", default=SUBSAMPLE,
                        choices=['naive_balancing', 'none'],
                        help="dose response subsample strategy; 'none' or 'naive_balancing'")
    parser.add_argument("--threads", default=THREADS,
                        help="number of threads per machine learning training job; -1 for using all threads")
    parser.add_argument("-o", "--out_dir", default=OUT_DIR,
                        help="output directory")
    return parser
