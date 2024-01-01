
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(3, 10)
        self.bn = torch.nn.BatchNorm2d(3)
        self.relu = torch.nn.ReLU()
    def forward(self, x3):
        x3 = self.linear1(x3)
        x3 = self.relu(self.bn(x3))
        return x3
# Inputs to the model
x3 = torch.randn(1, 3)
