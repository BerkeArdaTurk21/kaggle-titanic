# Titanic: Machine Learning from Disaster

End-to-end tabular ML project on the Kaggle
[Titanic](https://www.kaggle.com/competitions/titanic) competition:
predict which passengers survived the sinking (binary classification, scored by accuracy).

**Best result:** logistic regression with a leak-free *group survival* feature,
**0.841** 5-fold CV accuracy and **0.79425** on the Kaggle leaderboard (baseline without the feature: 0.76555).

**In short:** tuned Random Forest and XGBoost did not beat a plain logistic regression on the same features.
What moved the score was new information (how the passenger's travel group fared), computed so that no
label leaks into the evaluation.

## Notebooks

Run them in order; each one ends with a *Findings* section.

| Notebook | What it does |
|---|---|
| [`01_eda.ipynb`](notebooks/01_eda.ipynb) | Data quality, target balance, survival by gender, class, age, fare, family, title, cabin; feature plan |
| [`02_baseline.ipynb`](notebooks/02_baseline.ipynb) | Majority class, gender rule and logistic regression with a shared 5-fold CV and a leak-free `Pipeline`; error analysis |
| [`03_trees.ipynb`](notebooks/03_trees.ipynb) | Random Forest and XGBoost, default vs tuned with nested CV, depth curve, permutation importance |
| [`04_group_features.ipynb`](notebooks/04_group_features.ipynb) | Group survival feature: two leaky versions vs the honest one, effect on both models, test coverage check |
| [`05_shap.ipynb`](notebooks/05_shap.ipynb) | SHAP explanations of the XGBoost + group model on held-out folds: global ranking, group feature by gender, one passenger step by step |

## Results

5-fold stratified cross-validation on the 891 training passengers, the same folds for every model.
All learned preprocessing (age imputation by title, scaling, one-hot encoding, group survival rates) runs inside a
scikit-learn `Pipeline`, so it is fitted on the training folds only. Tree model hyperparameters are tuned with
nested cross-validation, so the reported score is not inflated by the search.

| Model | CV accuracy (mean ± std) | Train accuracy | Kaggle public score |
|---|---|---|---|
| Majority class ("everyone died") | 0.616 ± 0.002 | 0.616 | |
| Gender rule ("women survived") | 0.787 ± 0.019 | 0.787 | |
| Logistic regression | 0.831 ± 0.010 | 0.834 | 0.76555 |
| Random forest (default) | 0.822 ± 0.019 | 0.989 | |
| XGBoost (default) | 0.822 ± 0.036 | 0.975 | |
| Random forest (tuned) | 0.831 ± 0.013 | 0.895 | |
| XGBoost (tuned) | 0.832 ± 0.025 | 0.886 | 0.77033 |
| **Logistic regression + group survival** | **0.841 ± 0.016** | (inflated) | **0.79425** |
| XGBoost (tuned) + group survival | 0.837 ± 0.007 | (inflated) | 0.78947 |

The ± value is the standard deviation of the five fold scores (`numpy.std`, ddof=0). It shows how much a score
moves from fold to fold; it is not a confidence interval, because the folds share most of their training data.

![Model comparison](reports/figures/model_comparison.png)

The group survival feature is the survival rate of the *other* known passengers sharing a ticket or family
(surname + fare). It is built from the target, so it is fitted inside the Pipeline with leave-one-out on the
training rows. Train accuracy for these models includes each passenger's own label and is not comparable.

![Leaky vs honest evaluation](reports/figures/group_feature_comparison.png)

## Key findings

- Gender, ticket class and the title in the passenger's name are the strongest signals.
- Logistic regression beats the gender rule on every fold, with no sign of overfitting (train 0.834).
- Tree ensembles with default settings memorise the training data (train 0.98–0.99) and score lower on new
  passengers. Tuned, they tie with logistic regression: fold-by-fold differences go both ways and are smaller
  than the fold-to-fold noise.
- Trees did not fix the groups logistic regression gets wrong (women in 3rd class, men in 1st class). With these
  features ~83% looks like a ceiling; further gains need new information, not a more flexible model.
- On Kaggle the first two models score about 6 points below their CV accuracy (0.766 and 0.770). The 0.5-point
  gap between them is one or two passengers, well within the noise.
- New information moves the plateau: the group survival feature lifts logistic regression to 0.841 and is never
  worse on any fold. It mostly helps women in 3rd class whose family perished (accuracy 65% → 74%).
  On Kaggle the gain is larger than in CV: logistic regression goes from 0.766 to 0.794 (+2.9 points, about
  12 test passengers) and XGBoost from 0.770 to 0.789. The test set has known group members slightly more
  often than the validation folds (41% vs 37%), which fits a larger gain there.
- Leakage demo: computing the same feature carelessly gives 0.983 (own label included) or 0.844 (computed before
  CV). The second is only 0.3 points above the honest score, which is exactly why it would go unnoticed.
- SHAP on held-out folds (XGBoost + group): title and gender share the top spot (0.69 and 0.67 mean |SHAP|,
  log-odds), followed by class and group survival. When all known group members died, the feature lowers a woman's
  log-odds by 1.27 on average. Men whose group survived are close to a coin flip: boys survive 93%, adult men 30%,
  and the feature cannot tell which adult men did.

![Random forest depth curve](reports/figures/rf_depth_curve.png)

![SHAP beeswarm](reports/figures/shap_beeswarm.png)

## Limitations

- 891 training passengers: one passenger is worth ~0.56 points of CV accuracy per fold, so differences of a point
  or two between models are within the noise.
- The Kaggle score is measured on a small test set and was checked only a handful of times; it was used to confirm
  results, not to choose models.
- The group survival feature relies on passengers travelling together appearing in both train and test, which is
  specific to this dataset.

## Model card

| | |
|---|---|
| **Model** | Logistic regression in a scikit-learn `Pipeline` with the group survival feature (`make_preprocessor(group_survival=True)` in `src/titanic/features.py`) |
| **Task** | Predict whether a Titanic passenger survived (binary classification) |
| **Intended use** | A learning project and Kaggle competition entry. Not meant for decisions about real people. |
| **Training data** | Kaggle Titanic `train.csv`: 891 passengers, 38% survived. Not included in this repository. |
| **Inputs** | Age (imputed by title), log fare, missing-age and has-cabin flags, ticket class, gender, title, family type, port of embarkation, group survival rate and whether the group is known |
| **Evaluation** | Accuracy on 5-fold stratified CV (`StratifiedKFold(5, shuffle=True, random_state=42)`), the same folds for every model, plus the Kaggle public leaderboard |
| **Performance** | CV 0.841 ± 0.016 (fold std); Kaggle public 0.79425 |
| **Where it fails** | The groups the baseline already got wrong: women in 3rd class (accuracy 65% → 74% with the group feature) and men in 1st class |
| **Limitations** | See [Limitations](#limitations). The group feature only helps when a passenger's travel companions are in the data. |
| **Ethical note** | The data records who survived in 1912, when gender and ticket class decided much of the outcome. The model learns these historical patterns; they say nothing about people today. |

## Project structure

```
kaggle-titanic/
├── data/
│   ├── raw/            # train.csv, test.csv from Kaggle (not committed)
│   └── processed/
├── notebooks/          # numbered notebooks, run in order
├── reports/
│   ├── figures/        # figures used in this README
│   └── cv_results.csv  # CV scores of every honest model
├── src/titanic/
│   ├── data.py         # load_raw: reads the CSVs, renames Sex -> Gender
│   └── features.py     # feature engineering, TitleAgeImputer, GroupSurvivalEncoder, make_preprocessor
├── submissions/        # CSV files sent to Kaggle
├── requirements.txt    # direct dependencies, pinned
├── requirements.lock   # every installed package, pinned (pip freeze on Windows, Python 3.12)
└── .python-version     # 3.12
```

## How to run

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; on macOS/Linux: source .venv/bin/activate
pip install -r requirements.lock   # exact versions behind the results; requirements.txt has only the direct ones
python -m ipykernel install --user --name kaggle-titanic --display-name "Python (kaggle-titanic)"
```

Download the data from the
[competition data page](https://www.kaggle.com/competitions/titanic/data)
(you need to join the competition first) and put `train.csv` and `test.csv` into `data/raw/`.
Then open the notebooks in order with the `Python (kaggle-titanic)` kernel.
Notebooks 03 and 04 run nested cross-validation and take a few minutes each on a laptop CPU.

## License

Code is released under the [MIT License](LICENSE). The Titanic data belongs to Kaggle and is not included;
download it from the competition page.
