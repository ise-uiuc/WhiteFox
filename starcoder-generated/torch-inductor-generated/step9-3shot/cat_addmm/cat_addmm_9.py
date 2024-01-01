
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(0.8)
        self.dense1 = torch.nn.Linear(10, 20)
        self.dense2 = torch.nn.Linear(15, 16)
        self.concat = torch.nn.Linear(4, 8)

    def forward(self, x1, x2, x3):
        v1 = self.dense1(x1)
        v2 = self.dense2(x2)
        v3 = torch.cat([v1, v2], dim = 0)
        v4 = self.concat(v3)
        out = self.dropout(v4)
        return out

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10)
x2 = torch.randn(1, 15)
x3 = torch.randn(1, 10)
