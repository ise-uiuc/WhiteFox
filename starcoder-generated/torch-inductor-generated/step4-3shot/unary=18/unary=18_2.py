
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(7, 5, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = torch.sigmoid(v1)
        v2 = self.conv(x1)
        return nn.Sigmoid()(v2)
# Inputs to the model
x1 = torch.randn(1, 7, 64, 64)
