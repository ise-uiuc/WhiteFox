
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(15, 1, 3, stride=2, padding=1)
    def forward(self, x1):
        v4 = self.conv(x1)
        v1 = v4 - 1.5707963267948966
        v2 = v1 - 0.12
        return v2
# Inputs to the model
x1 = torch.randn(1, 15, 125, 125)
