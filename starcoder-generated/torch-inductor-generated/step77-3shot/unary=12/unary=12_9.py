
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(64, 128, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = torch.sigmoid(x1)
        v2 = self.conv(v1)
        v3 = torch.sigmoid(v2)
        return v2 * v3
# Inputs to the model
x1 = torch.randn(1, 64, 64, 64)
