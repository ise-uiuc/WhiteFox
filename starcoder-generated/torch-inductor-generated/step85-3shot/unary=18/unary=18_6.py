
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(16, 18)
        self.l2 = torch.nn.Linear(18, 24)
        self.l3 = torch.nn.Linear(24, 20)
        self.l4 = torch.nn.Linear(20, 16)
        self.l5 = torch.nn.Linear(16, 12)
    def forward(self, x1):
        v1 = self.l1(x1)
        v2 = torch.relu(v1)
        v3 = self.l2(v2)
        v4 = torch.relu(v3)
        v5 = self.l3(v4)
        v6 = torch.relu(v5)
        v7 = self.l4(v6)
        v8 = torch.relu(v7)
        v9 = self.l5(v8)
        v10 = torch.relu(v9)
        return v10
# Inputs to the model
x1 = torch.randn(1, 16)
