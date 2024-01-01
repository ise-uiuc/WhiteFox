
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(24, 48)
        self.linear1 = torch.nn.Linear(48, 96)
        self.linear2 = torch.nn.Linear(96, 144)
        self.linear3 = torch.nn.Linear(144, 275)
    def forward(self, x2):
        v1 = self.linear(x2)
        v2 = torch.tanh(v1)
        v3 = v2 * 0.203133
        v4 = v2 * v2
        v5 = v4 * v2
        v6 = v5 * 0.043815
        v7 = v2 + v6
        v8 = v7 * 1.4884692569085928
        v9 = v8 + 1
        v10 = v3 * v9
        return v10
# Inputs to the model
x2 = torch.randn(1, 3, 4, 4)
