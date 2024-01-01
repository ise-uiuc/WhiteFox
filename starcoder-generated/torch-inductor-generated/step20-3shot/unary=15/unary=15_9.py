
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn1 = torch.nn.BatchNorm1d(160)
        self.fc1 = torch.nn.Linear(160, 8)
        self.relu1 = torch.nn.ReLU()
        self.bn2 = torch.nn.BatchNorm1d(8)
        self.fc2 = torch.nn.Linear(8, 80)
    def forward(self, x1):
        v1 = self.bn1(x1)
        v2 = self.fc1(v1)
        v3 = self.relu1(v2)
        v4 = self.bn2(v3)
        v5 = self.fc2(v4)
        return v5
# Inputs to the model
x1 = torch.randn(4, 160)
