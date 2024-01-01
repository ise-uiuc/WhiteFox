
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 1, 3, stride=1, padding=1)
        self.bn = torch.nn.BatchNorm2d(1)
    def forward(self, x1):
        v1 = self.conv1(x1) # Apply pointwise convolution with kernel size 3 and stride 1 on x1, and record the result as v1
        v2 = self.bn(self.conv2(v1)) # Apply pointwise convolution with kernel size 3 and stride 1 on v1 and then feed the output to BatchNorm2d, and record the result as v2
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 36, 36)
