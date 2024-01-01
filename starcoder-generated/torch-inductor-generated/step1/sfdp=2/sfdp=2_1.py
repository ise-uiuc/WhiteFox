
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dense1 = torch.nn.Dense(64, 64)
        self.dense2 = torch.nn.Dense(64, 32)
 
    def forward(self, x, y):
        v1 = self.dense1(x)
        v2 = self.dense2(y)
        v3 = torch.matmul(v1, y.transpose(-2, -1))
        v4 = torch.max(v3)
        v5 = v3 / v4
        v6 = torch.dropout(v5, training=True, p=0.2)
        v7 = torch.matmul(v6, v2)
        return v7

# Initializing the model
m = Model()

# Inputs to the model
x, y = torch.randn(32, 64), torch.randn(32, 32)
