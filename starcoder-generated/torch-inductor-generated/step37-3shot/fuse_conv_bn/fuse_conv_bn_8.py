
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(1, 4)
        self.bn = torch.nn.BatchNorm1d(4)
    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        return x
# Inputs to the model
x = torch.randn(1, 1)
