
class Model(torch.nn.Module):
    def __init__(self, nhead):
        super().__init__()
        self.nhead = nhead

    def forward(self, query, key, value, dropout_p=0.):
        scale_factor = torch.tensor(1 / math.sqrt(self.nhead), dtype=torch.float32)
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        output = torch.nn.functional.dropout(softmax_qk, p=dropout_p).matmul(value)
        return output

# Initializing the model
n = 4
m = Model(n)

# Inputs to the model
query = torch.randn(1, 4, 300)
key = torch.randn(1, 4, 400)
value = torch.randn(1, 4, 400)
