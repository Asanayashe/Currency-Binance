import requests
import pandas as pd
import torch
import torch.nn as nn
from model import Model
from sklearn.preprocessing import StandardScaler
import torch.optim as optim
import matplotlib.pyplot as plt
from dataset import CryptoDataset
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae


def get_data(coin: str, interval: str = '1h', limit: int = 500):
    url = 'https://api.binance.com/api/v3/klines'
    params = {'symbol': coin, 'interval': interval, 'limit': limit}
    resp = requests.get(url, params)
    data = resp.json()
    for idx, item in enumerate(data):
        data[idx] = list(map(float, item))
    columns = ["start_time", "open", "high", "low", "close", "volume", "close_time", "quote_asset_volume",
               "number_of_trades", "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"]
    df = pd.DataFrame(data, columns=columns)
    df["start_time"] = pd.to_datetime(df["start_time"], unit='ms')
    df["close_time"] = pd.to_datetime(df["close_time"], unit='ms')
    df.to_csv('coins.csv', index=False)
    return df


def train_model(model, loader, optimizer, criterion):
    n_epoch = 100
    history= []
    for _ in range(n_epoch):
        for x, y in loader:
            optimizer.zero_grad()
            preds = model(x.permute(0, 2, 1))
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            history.append(loss.detach().numpy())
    return history


def test_model(model, loader):
    model.eval()
    predictions = []
    true_values = []
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.permute(0, 2, 1))
            pred = scaler.inverse_transform(pred.reshape(-1, 1))
            predictions.extend(pred.reshape(-1))
            y = scaler.inverse_transform(y.reshape(-1, 1))
            true_values.extend(y.reshape(-1))
    print("MAE:", mae(true_values, predictions))
    print("MSE:", mse(true_values, predictions))
    return predictions


def predict_next_price(model, values):
    values = values.reshape(1, -1, 1)
    print(values.shape)
    answer = model(values.permute(0, 2, 1))
    answer = scaler.inverse_transform(answer.detach().numpy().reshape(1, -1))
    print('New price:', answer)
    values = values.reshape(-1).tolist()
    answer = answer.reshape(-1).tolist()
    plt.plot(values + answer)
    plt.plot(seq_len, answer, 'ro')
    plt.ylabel('Price')
    plt.xlabel('Data')
    plt.title('Next price prediction')
    plt.show()


if __name__ == "__main__":
    df = get_data("BTCUSDT", interval='1h', limit=1000)
    train_len = int(len(df)*0.7)
    data = df.filter(['close'])
    x = df.filter(['close']).values

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(x)
    train_data = scaled_data[:train_len]

    seq_len = 48
    batch_size = 32
    train_dataset = CryptoDataset(train_data, seq_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = Model(input_size=seq_len, hidden_size=256, n_layers=4, dropout=0.2)
    model.train()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    history = train_model(model, train_loader, optimizer, criterion)
    plt.plot(history)
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Loss')
    plt.show()

    test_data = scaled_data[train_len - seq_len:, :]
    test_dataset = CryptoDataset(test_data, seq_len)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    preds = test_model(model, test_loader)

    train = data[:train_len]
    valid = data[train_len:]
    valid['Predictions'] = preds

    plt.figure(figsize=(16, 8))
    plt.title('Cryptocurrency rate')
    plt.xlabel('Data')
    plt.ylabel('Price')
    plt.plot(train['close'])
    plt.plot(valid[['close', 'Predictions']])
    plt.legend(['Train', 'Test', 'Predictions'], loc='lower right')
    plt.show()

    values = valid.filter(['close']).values[len(valid) - seq_len:]
    values = torch.Tensor(values)
    predict_next_price(model, values)
