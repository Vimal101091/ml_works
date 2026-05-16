# NSE Stock Price Predictor

A deep learning-based stock price prediction system for NSE (National Stock Exchange of India) stocks using LSTM neural networks.

## Features

- **Comprehensive Stock Coverage**: Supports 200+ NSE stocks
- **LSTM Deep Learning Model**: Bidirectional LSTM architecture for accurate predictions
- **GPU Optimized**: Optimized for Google Colab with GPU support
- **REST API**: Get predictions via HTTP endpoints
- **Model Persistence**: Save and load trained models

## Requirements

- Python 3.8+
- TensorFlow
- pandas
- numpy
- scikit-learn
- yfinance
- flask
- flask-cors
- matplotlib

## Installation

```bash
pip install yfinance tensorflow scikit-learn pandas numpy matplotlib flask flask-cors
```

## Usage

### Training the Model

1. Open `stock_predictor/nifty50_lstm_all_nse.ipynb` in Google Colab
2. Run all cells to train the model
3. Model and data will be saved to Google Drive

### Making Predictions

```python
from stock_predictor import predict_stock

result = predict_stock('RELIANCE')
print(result)
# Output: {'symbol': 'RELIANCE.NS', 'last': 2500.0, 'predicted': 2520.5, 'change_pct': 0.82}
```

### Running the REST API

```python
from stock_predictor import run_api
run_api(port=5000)
```

API Endpoints:
- `GET /predict/<symbol>` - Get prediction for a stock
- `GET /stocks` - List all available stocks
- `GET /health` - Health check

## Model Architecture

- Bidirectional LSTM (128 units)
- Dropout (0.2)
- Bidirectional LSTM (64 units)
- Dropout (0.2)
- LSTM (32 units)
- Dropout (0.2)
- Dense (64 units, ReLU)
- Dense (output size)

## Configuration

Key parameters in the notebook:
- `STOCK_LIMIT`: Number of stocks to train (default: 100)
- `SEQUENCE_LENGTH`: Input sequence length (default: 60 days)
- `EPOCHS`: Training epochs (default: 50)
- `BATCH_SIZE`: Batch size (default: 64)
- `TEST_SPLIT`: Train/test split ratio (default: 0.2)

## Output Files

After training, the following files are saved:
- `model.h5` - Trained LSTM model
- `scaler_x.pkl` - Input feature scaler
- `scaler_y.pkl` - Output scaler
- `data.pkl` - Training data
- `meta.json` - Model metadata

## License

MIT License