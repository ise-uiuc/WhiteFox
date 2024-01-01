
class Model(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.linear1 = torch.nn.Linear(width, width)
        self.linear2 = torch.nn.Linear(width, width)
        self.bn1 = torch.nn.BatchNorm1d(width)
        self.bn2 = torch.nn.BatchNorm1d(width)
    def forward(self, x):
        v1 = self.bn1(self.linear1(x))
        v2 = self.bn2(self.linear2(x))
        return v1
# Inputs to the model
x1 = torch.randn(10, 1)
