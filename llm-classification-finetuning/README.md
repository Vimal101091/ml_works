# LLM Classification Finetuning

A machine learning project for classifying LLM competition outputs. The goal is to predict which model wins in head-to-head comparisons between LLMs.

## Classes

- `winner_model_a` - Model A wins
- `winner_model_b` - Model B wins
- `winner_tie` - Tie

## Approach

### Baseline (Current)
- TF-IDF features from `prompt`, `response_a`, `response_b`
- Length features (word count, character count)
- Logistic Regression classifier

### Future (Planned)
- BERT embeddings
- LoRA fine-tuning

## Requirements

- pandas
- numpy
- scikit-learn
- scipy

## Files

- `01_logistic_regression_baseline.ipynb` - Baseline model
- `train.csv` - Training data
- `test.csv` - Test data
- `sample_submission.csv` - Submission format

## Usage

Run the notebook in Kaggle or locally to generate predictions.

```bash
# Install dependencies
pip install pandas numpy scikit-learn scipy
```

## License

MIT License