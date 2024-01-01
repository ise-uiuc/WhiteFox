
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Sequential(torch.nn.Conv2d(16, 32, 3), torch.nn.BatchNorm2d(32), torch.nn.ReLU(inplace=False), torch.nn.AdaptiveMaxPool2d([1, None]))
    def forward(self, x):
        return self.conv(x)
# Inputs to the model
x = torch.randn(1, 16, 32, 32)
