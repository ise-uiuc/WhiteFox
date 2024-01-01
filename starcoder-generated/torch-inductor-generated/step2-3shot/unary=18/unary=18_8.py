
class Model(torch.nn.Module):
    def __init__(self):
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        super().__init__()
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.sigmoid(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
