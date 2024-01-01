
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1.sigmoid()
        return v1 - 0.5
# Inputs to the model
x1 = torch.randn(1, 3, 65, 65)
