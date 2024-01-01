
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(128, 64)
        self.linear2 = torch.nn.Linear(64, 32)
        self.linear3 = torch.nn.Linear(32, 8)
        self.linear4 = torch.nn.Linear(8, 1)
    def forward(self, x1):
        v1 = self.linear1(x1)
        v2 = F.relu(v1)
        v3 = self.linear2(v2)
        v4 = F.relu(v3)
        v5 = self.linear3(v4)
        v6 = F.relu(v5)
        v7 = self.linear4(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 128)
