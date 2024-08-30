from torch import nn


class LSTMPredictor(nn.Module):
    def __init__(self, config):
        super(LSTMPredictor, self).__init__()
        input_size = config["cae"]["latent_dim"]
        hidden_size = config["cae_lstm"]["hidden_dim"]
        self.num_conditional = config["cae_lstm"]["num_conditional"]

        self.lstm = nn.LSTM(
            # input_size=input_size,
            input_size=input_size + self.num_conditional,  # conditional
            hidden_size=hidden_size,
            num_layers=5,
            dropout=0.5,
            batch_first=True,
        )

        self.linear = nn.Linear(hidden_size, input_size * 3)

    def forward(self, x):
        batch_size, fragment_length, latent_dim_AE = x.shape
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.linear(out)
        # out: (batch_size, latent_dim_AE, hidden_size)
        # out = out.reshape(batch_size, fragment_length, latent_dim_AE)
        out = out.reshape(batch_size, fragment_length, latent_dim_AE - self.num_conditional)  # conditional
        return out
