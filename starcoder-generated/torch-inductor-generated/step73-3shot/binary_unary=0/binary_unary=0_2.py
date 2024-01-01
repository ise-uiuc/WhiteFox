
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(16, 16)
        self.linear2 = torch.nn.Linear(16, 16)
        self.linear3 = torch.nn.Linear(16, 16)
    def forward(self, x):
        v1 = self.linear1(x)
    def forward(self, x, y):
        v1 = self.linear1(x)
        v2 = self.linear2(x)
        v3 = self.linear1(x)
        v4 = v1 + v2
        v5 = v3
        v6 = self.linear3(v1) + v1
        v7 = v5
        v8 = v5
        return v2
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
