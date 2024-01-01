
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 1, 3, stride=2, padding=2, dilation=2)
    def forward(self, x):
        x = self.conv(x)
        x = torch.tanh(x)
        return x
x = torch.randn(1, 3, 224, 224)
