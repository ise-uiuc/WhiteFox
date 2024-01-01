
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1024, 512, 3, stride=2, padding=1)
        self.relu = torch.nn.ReLU()
        self.bn = torch.nn.BatchNorm2d(512)
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
# Inputs to the model
x = torch.randn(1, 1024, 14, 14)
