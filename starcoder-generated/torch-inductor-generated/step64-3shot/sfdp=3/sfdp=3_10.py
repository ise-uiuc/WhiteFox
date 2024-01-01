
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value, scale_factor, dropout_p):
        out = torch.matmul(query, key.transpose(-2, -1))
        out = out.mul(scale_factor)
        out = out.softmax(dim=-1)
        out = torch.nn.functional.dropout(out, p=dropout_p)
        out = out.matmul(value)
        return out

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 32, 8)
key = torch.randn(1, 64, 16)
value = torch.randn(1, 64, 16)
scale_factor = 1.0
dropout_p = 0.5
