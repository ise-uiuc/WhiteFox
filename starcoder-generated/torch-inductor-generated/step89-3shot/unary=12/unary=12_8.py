
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = torch.nn.MaxPool2d((2, 2), stride=2, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        self.conv = torch.nn.Conv2d(64, 32, 1, stride=1)
        self.mul = torch.mul
    def forward(self, x1):
        v1 = self.maxpool(x1)
        v2 = self.conv(v1)
        v3 = self.mul(x1, v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 64, 28, 28)
