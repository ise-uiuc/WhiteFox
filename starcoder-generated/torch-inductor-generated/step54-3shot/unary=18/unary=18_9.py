
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 7, stride=1, padding=3, dilation=3)
    def forward(self, x1):
        return self.conv(x1)
# Inputs to the model
x1 = torch.randn(1, 3, 40, 60)
