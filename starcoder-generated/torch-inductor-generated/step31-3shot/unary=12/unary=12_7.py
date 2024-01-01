
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 7, 6, stride=1, padding=4)
    def forward(self, x1):
        v1 = self.sigmoid(x1)
        v2 = self.conv(v1)
        return v2.sigmoid()
# Inputs to the model
x1 = torch.randn(1, 2, 64, 64)
