
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(64, 64, (3,3))
        self.bn = torch.nn.BatchNorm2d(64)
    def forward(self, x):
        x = torch.nn.functional.relu_(self.conv(x))
        x = torch.nn.functional.relu(self.bn(x))
        return x
# Inputs to the model
x = torch.randn(1, 64, 16, 16)
