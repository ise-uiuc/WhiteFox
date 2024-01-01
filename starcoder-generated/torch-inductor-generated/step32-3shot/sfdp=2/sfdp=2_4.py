
class MultiHeadAttention(nn.Module):
    def __init__(self,
                 heads: int,
                 d_model: int,
                 dropout_probability: float):
        super().__init__()

        self.mha = nn.MultiheadAttention(d_model, heads, dropout=dropout_probability, batch_first=True)
        self.drop = nn.Dropout(dropout_probability)
        self.identity = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.identity(self.mha(x, self.drop(x), self.drop(x))[0])

def get_model(dropout_p: float):
    return MultiHeadAttention(1, 224, dropout_p)

m = get_model(0.0)

# Inputs to the model
x_ = torch.randn(1, 196, 224)
