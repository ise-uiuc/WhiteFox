
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(16, 16)
        self.conv = torch.nn.Conv3d(16, 16, (3, 3, 3))
        self.bn = torch.nn.BatchNorm3d(16)
    def forward(self, x):
        y = self.fc(x)
        y = self.conv(y)
        y = self.bn(y)
        return y
# Inputs to the model
x = torch.randn(1, 16)
