
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv3d(1, 1, 3)
        self.bn = torch.nn.BatchNorm3d(1)
        self.relu = torch.nn.ReLU()
        self.pool3d = torch.nn.MaxPool3d(2)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool3d(x)
        return x
# Inputs to the model
x = torch.randn(1, 1, 4, 4, 4)
