
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query = torch.nn.Linear(32, 32)
        self.key = torch.nn.Linear(32, 32)

    def forward(self, x1, x2):
        v1 = self.query(x1)
        v2 = self.key(x2)
        qk = torch.matmul(v1, v2.transpose(-2, -1))
        v4 = qk.mul(0.125)
        v3 = v4.softmax(dim=-1)
        v5 = torch.nn.functional.dropout(v3, 0.5)
        v6 = v5.matmul(x2)
        return v6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 32)
x2 = torch.randn(1, 32)
