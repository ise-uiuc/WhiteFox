
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout_p = 0.5
        self.attn = torch.nn.MultiheadAttention(8, 2, dropout=self.dropout_p)
    def forward(self, x1, x2):
        v1 = self.attn(x1, x2)
        return v1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(16, 8, 64)
x2 = torch.randn(16, 16, 64)
