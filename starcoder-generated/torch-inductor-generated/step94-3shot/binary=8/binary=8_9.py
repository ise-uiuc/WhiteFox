
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(3, 32, 5, stride=1, padding=2)
    def forward(self, x1, x2, x3):
        v1 = self.conv1(x1) # pointwise convolution
        v2 = self.conv2(x2)
        v3 = self.conv2(x3)
        v4 = v1 + v2 + v3
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 3, 3)
x2 = torch.randn(1, 3, 3, 3)
x3 = torch.randn(1, 3, 3, 3)
