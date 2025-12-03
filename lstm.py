import torch.nn as nn

class StockLSTM(nn.Module):
    """
    Direct Hochreiter & Schmidhuber (1997) LSTM
    """

    def __init__(self, input_size=1, hidden_size=64, num_layers=2, drop_out=0.2):
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=drop_out)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])  # Use last time step output
