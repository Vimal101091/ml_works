# Titanic Survival Prediction

Machine learning project to predict passenger survival on the Titanic.

## Data Source

Data from Kaggle: https://www.kaggle.com/competitions/titanic

## Approach

- Feature engineering: Title extraction, family size, deck, age groups
- Multiple models tested: GradientBoosting, ExtraTrees, LogisticRegression, XGBoost, MLP, GMM Kernel
- Ensemble methods

## Key Features

- Pclass, Sex, Age, SibSp, Parch, Fare
- Title (Mr, Miss, Mrs, Master, etc.)
- FamilySize, IsAlone
- HasCabin, Deck
- IsChild, IsWomanOrChild

## Requirements

- pandas
- numpy
- scikit-learn
- xgboost (optional)

## Files

- `titanic_final.ipynb` - Final model
- `titanic_pipeline.py` - Reusable pipeline
- `titanic_model.py` - Model module
- `gender_submission.csv` - Baseline submission

## Usage

Download data from Kaggle and run the notebooks to generate predictions.

## License

MIT License