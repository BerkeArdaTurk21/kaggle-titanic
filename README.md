# Titanic: Machine Learning from Disaster

End-to-end tabular ML project on the Kaggle
[Titanic](https://www.kaggle.com/competitions/titanic) competition:
predict which passengers survived the sinking (binary classification, scored by accuracy).

> Work in progress. This README is filled in as each step is finished.

## Approach

| Step | Status | Where |
|---|---|---|
| 1. Exploratory data analysis | done | `notebooks/01_eda.ipynb` |
| 2. Baseline (logistic regression) | planned | |
| 3. Tree models (Random Forest, XGBoost) | planned | |
| 4. Cross-validation and model selection | planned | |
| 5. Kaggle submission | planned | `submissions/` |

## Results

_Coming after step 4._

| Model | CV accuracy (mean ± std) | Kaggle public score |
|---|---|---|
| Majority class ("everyone died") | | |

## Project structure

```
kaggle-titanic/
├── data/
│   ├── raw/            # train.csv, test.csv from Kaggle (not committed)
│   └── processed/
├── notebooks/          # numbered notebooks, run in order
├── reports/figures/    # figures used in this README
├── src/                # reusable code (features, pipelines)
├── submissions/        # CSV files sent to Kaggle
└── requirements.txt
```

## How to run

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Download the data from the
[competition data page](https://www.kaggle.com/competitions/titanic/data)
(you need to join the competition first) and put `train.csv` and `test.csv` into `data/raw/`.
Then open the notebooks in order and select the `.venv` interpreter.
