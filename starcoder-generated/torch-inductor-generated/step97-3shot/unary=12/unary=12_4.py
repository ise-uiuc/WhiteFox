
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 16, 1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1.sigmoid()
        v3 = torch.sigmoid(v2)
        v4 = v2 + v3
        return v4
# Inputs to the model
x1 = torch.randn(1, 1, 224, 224)
