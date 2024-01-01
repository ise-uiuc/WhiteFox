
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(3, 5, 2, stride=3)
        self.bn = torch.nn.BatchNorm1d(5, affine=True)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.bn(x)
        return x
# Inputs to the model
x = torch.randn(1, 3, 6)
