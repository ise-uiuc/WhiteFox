
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 192, 3, padding=1, stride=1, dilation=1, groups=1, bias=True)
    def forward(self, x1):
        v1 = self.conv(x1)
        h = torch.relu(v1)
        return h
# Inputs to the model
x1 = torch.randn(1, 16, 4, 4)
