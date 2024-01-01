
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(16, 32)
        self.linear2 = torch.nn.Linear(32, 48)
    def forward(self, x1):
        v1 = self.linear2(self.linear1(x1))
        v2 = self.linear1(x1)
        v3 = self.linear2(v2)
        v4 = self.linear1(x1)
        v5 = self.linear2(v4)
        v6 = self.linear1(x1)
        v7 = self.linear2(v6)
        v8 = v1 + v3 + v5 + v7
        v9 = torch.relu(v8)
        return v9
# Inputs to the model
x1 = torch.randn(1, 16)
