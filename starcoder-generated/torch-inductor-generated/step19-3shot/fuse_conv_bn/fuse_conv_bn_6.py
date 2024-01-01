
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 2, 2)
        self.bn = torch.nn.BatchNorm1d(2)
        self.linear_relu = torch.nn.Sequential(
            torch.nn.Linear(2, 3),
            torch.nn.ReLU(),
        )
    def forward(self, x):
        y = self.conv(x)
        y = self.linear_relu(y)
        z = self.bn(y)
        return t.sum(z)
# Inputs to the model
x = torch.randn(1, 2, 4, 4)
