
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1, stride=1, padding=0, groups=3)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x1):
        x1 = self.conv(x1) / 3
        x2 = torch.clip(x1, 0, 6)
        x3 = self.sigmoid(x2)
        return x3
# Inputs to the model
x_1 = torch.randn(1, 3, 256, 256)
