
class Model(torch.nn.Module):
    def __init__(self, d1, m1, d2, m2):
        super().__init__()
        self.dense1 = torch.nn.Linear(d1, m1)
        self.dense2 = torch.nn.Linear(d2, m2)

    def forward(self, query, key, value, dropout_p):
        v1 = self.dense1(query)
        v2 = self.dense2(key)
        qk = torch.matmul(v1, v2.transpose(-2,-1))
        scale_factor = 1 / (v1.size(-1) ** 0.25)
        sotmax_qk = qk * scale_factor
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model(32, 8, 32, 16)

# Inputs to the model
query = torch.randn(1, 3, 64, 64)
key = torch.randn(1, 3, 64, 64)
value = torch.randn(1, 3, 64, 64)
dropout_p = torch.nn.Parameter(torch.arange(0.0, 0.1, 0.01))
