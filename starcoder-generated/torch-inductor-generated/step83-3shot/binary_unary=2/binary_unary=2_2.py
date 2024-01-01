
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(3, 8, bias=True)
        self.linear2 = torch.nn.Linear(3, 16, bias=True)
        self.linear3 = torch.nn.Linear(3, 32, bias=True)
        self.linear4 = torch.nn.Linear(3, 64, bias=True)
    def forward(self, x1):
        v1 = self.linear1(x1)
        v2 = v1 - 0.5
        v3 = F.relu(v2)
        v5 = self.linear2(x1)
        v6 = v5 - -0.5
        v7 = F.relu(v6)
        v9 = self.linear3(x1)
        v10 = v9 - -1.5
        v11 = F.relu(v10)
        v13 = self.linear4(x1)
        v14 = v13 - -3.0
        v15 = F.relu(v14)
        return v15
# Inputs to the model
x1 = torch.randn(3, 3)
