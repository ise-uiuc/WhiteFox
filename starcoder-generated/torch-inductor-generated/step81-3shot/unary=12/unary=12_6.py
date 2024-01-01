
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(128, 128, 1, stride=1, padding=0)
        self.conv_1 = torch.nn.Conv2d(128, 128, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv(x1)
        v1 = v1 + v1
        v2 = self.conv_1(v1)
        v2 = v2 + v1
        return v2
# Inputs to the model
x1 = torch.randn(1, 128, 32, 32)
