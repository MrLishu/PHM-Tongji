import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=6,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.05,
        )
        self.out = nn.Sequential(
            nn.Linear(128, 10),
            nn.BatchNorm1d(10),
            nn.ReLU(),
            nn.Linear(10, 1),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1).contiguous()
        # print("\n\n------\n\n")
        # print(x.shape)
        r_out, (h_n, h_c) = self.lstm(x, None)
        # print(r_out.shape)
        out = self.out(r_out[:, -1, :])
        # out = out.squeeze(-1)
        # print(out.shape)
        return out
