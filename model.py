import torch.nn as nn


class Model(nn.Module):

    def __init__(self, input_size: int, hidden_size: int, n_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=n_layers, batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_size, 100)
        self.ac = nn.ReLU()
        self.fc2 = nn.Linear(100, 50)
        self.dropout = nn.Dropout(dropout)
        self.fc3 = nn.Linear(50, 1)

    def forward(self, x):
        out, (h, c) = self.lstm(x)
        out = self.ac(self.fc1(out))
        out = self.ac(self.fc2(out))
        out = self.dropout(out)
        out = self.fc3(out)
        return out
