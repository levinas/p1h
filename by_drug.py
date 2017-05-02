from __future__ import print_function

import os

from datasets import NCI60
from regression import regress
from argparser import get_parser


def test1():
    df = NCI60.load_by_drug_data()
    regress('XGBoost', df)


def test2():
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=20)
    df = NCI60.load_by_drug_data()
    regress(model, df, cv=2)


def main():
    description = 'Build ML models to predict by-drug tumor response.'
    parser = get_parser(description)
    args = parser.parse_args()

    print('Args:', args, end='\n\n')
    if args.use_gi50:
        print('Use NCI GI50 value instead of percent growth')
    else:
        print('Use percent growth at log concentration {}'.format(args.logconc))
    print()

    # cells = NCI60.all_cells() if 'all' in args.cells else args.cells
    drugs = args.drugs

    for drug in drugs:
        print('-' * 10, 'Drug NSC:', drug, '-' * 10)
        df = NCI60.load_by_drug_data(drug, cell_features=args.cell_features, scaling=args.scaling,
                                     use_gi50=args.use_gi50, logconc=args.logconc,
                                     subsample=args.subsample, feature_subsample=args.feature_subsample)
        if not df.shape[0]:
            print('No response data found')
            continue
        for model in args.models:
            regress(model, df, cv=args.cv, threads=args.threads, prefix=os.path.join(args.out_dir, 'NSC_' + drug))


if __name__ == '__main__':
    main()
