
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 10, 3, stride=2, padding=2, dilation=1)
    def forward(self, x1):
        v1 = F.relu(self.conv(x1))
        v2 = self.conv(x1)
        return v1, v2
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
