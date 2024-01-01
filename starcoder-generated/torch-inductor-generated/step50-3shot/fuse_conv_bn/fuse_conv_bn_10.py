
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(3,5)
        self.fc2 = torch.nn.Linear(5,6)
        self.fc3 = torch.nn.Linear(6,7)
        self.bn1 = torch.nn.BatchNorm1d(3)
        self.bn2 = torch.nn.BatchNorm1d(5)
        self.bn3 = torch.nn.BatchNorm1d(2)
    def forward(self, x1):
        s = self.fc1(x1)
        t = self.fc2(s)
        y = self.bn2(t)
        z1 = self.bn1(x1)
        z2 = self.fc3(z1)
        return y
# Inputs to the model
x = torch.randn(1, 3)
