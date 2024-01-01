
class Conv2d_maxpool(torch.nn.Module):
    def __init__(self, maxpool):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(16, 16, 1, stride=1, padding=0)
        self.maxpool2d = torch.nn.MaxPool2d(maxpool)
    def forward(self, x0):
        v0 = self.conv2d(x0)
        v1 = self.maxpool2d(v0)
        return v1
maxpool = 3
# Inputs to the model
x0 = torch.randn(1, 16, 10, 10)
