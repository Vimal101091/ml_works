# Titanic Survival Prediction

Machine learning project to predict passenger survival on the Titanic.

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
- `train.csv` - Training data
- `test.csv` - Test data
- `gender_submission.csv` - Baseline submission

## Usage

Run any notebook to generate predictions for the competition.

## License

MIT License