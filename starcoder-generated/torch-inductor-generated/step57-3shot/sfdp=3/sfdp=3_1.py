
class Model(torch.nn.Module):
    def __init__(self, n, dim, dropout):
        super(Model, self).__init__()
        self.proj = torch.nn.Linear(n, dim, bias=False)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x1):
        q = self.proj(x1)
        k = self.proj(x1)
        v = self.proj(x1)
        scale = float(dim) ** -0.5
        b = torch.matmul(q, k.transpose(-2, -1))
        scaled_b = b * scale
        softmax_b = torch.nn.functional.softmax(scaled_b, dim=-1)
        dropout_b = self.dropout(softmax_b)
        y = dropout_b.matmul(v)
        return y

# Initializing the model
n = 8
dim = 8
dropout = 0.5
m = Model(n, dim, dropout)

# Inputs to the model
x1 = torch.randn(1, 4, n)
