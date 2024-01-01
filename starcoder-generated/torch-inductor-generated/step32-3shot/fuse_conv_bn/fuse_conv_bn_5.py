
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(None, None, 3) # None for in_channels and out_channels
        self.bn = torch.nn.BatchNorm2d(None) # None for num_features
    def forward(self, x):
        x1 = self.conv1(x)
        y2 = self.bn(x1)
        return y2
# Inputs to the model
x1 = torch.randn(1, 3, 3, 3)
