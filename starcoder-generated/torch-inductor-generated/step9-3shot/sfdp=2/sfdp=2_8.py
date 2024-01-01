
class DotProductAttention(object):
    def __init__(self, dropout = 0):
        super().__init__()
        self.dropout = dropout
    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        softmax_qk = F.softmax(qk, dim = -1)
        dropout_qk = F.dropout(softmax_qk, self.dropout)
        output = torch.matmul(dropout_qk, value)
        return output

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = DotProductAttention()
 
    def forward(self, x1):
        e1 = self.attention(x1, x1, x1)
        e2 = torch.nn.functional.softmax(torch.matmul(x1, x1.transpose(-2, -1)), dim = -1)
        e3 = torch.nn.functional.linear(e2, e2)
        e4 = e3 + e2
        return e4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 10, 10)
v1 = m(x1)

# Expected output
v1_expected = torch.matmul(x1, x1.transpose(-2, -1)).softmax(dim = -1)
assert((v1 - v1_expected).abs().mean() < 1e-10)