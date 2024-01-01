
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(1, 1, 1)
        self.bn1 = torch.nn.BatchNorm1d(1)
        self.fc = torch.nn.Linear(1, 1)
    def forward(self, y):
        a = self.conv1(y)
        b = self.bn1(a)
        c = self.fc(b)
        return c
# Inputs to the model
y = torch.randn(1, 1)
