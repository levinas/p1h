## Pilot1 Hackathon

Python functions and command line utilities for working with standard pilot1 data sets.

### Command-line Examples

#### Export by-drug data
Save molecular features and dose reponse data for given drugs to CSV files:
```
python dataframe.py --by drug --drugs 100071 --feature_subsample 10
```

#### Export by-cell data
Save drug features and dose response data for given cell lines to CSV files:
```
python dataframe.py --by cell --cells BR:MCF7 CNS:U251
```

### Code Examples

#### Run standard regressions on a drug

```
from datasets import NCI60
from skwrapper import regress

df = NCI60.load_by_drug_data(drug='100071')
regress('XGBoost', df)
regress('Lasso', df)
```

#### Sweep customized RandomForest regression on all cell lines
```
from datasets import NCI60
from skwrapper import regress
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=20)

cells = NCI60.all_cells()
for cell in cells:
    df = NCI60.load_by_cell_data(cell)
    regress(model, df, cv=3)
```	



