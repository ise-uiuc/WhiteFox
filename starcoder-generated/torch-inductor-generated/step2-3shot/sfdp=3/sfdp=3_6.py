
class Model(torch.nn.Module):
    def __init__(self, hidden_size, num_hidden_layers, num_attention_heads):
        super().__init__()
        self.encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                hidden_size,
                num_attention_heads,
                hidden_dropout=0.0,
                batch_first=True
            ),
            num_hidden_layers,
        )

    def forward(self, x):
        v = self.encoder(x)
        return v

# Initializing the model
m = Model(num_hidden_layers=2, num_attention_heads=2, hidden_size=64)

# Input to the model
x = torch.randn(3, 3, 64)  # sequence, batch, feature
