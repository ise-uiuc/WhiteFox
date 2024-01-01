
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(4, 4, 31, stride=2, padding=2, dilation=1, groups=1, bias=True)
    def forward(self, x):
        x = self.conv2d(x)
        return x
# Inputs to the model
x = torch.randn(2, 4, 32, 32)
