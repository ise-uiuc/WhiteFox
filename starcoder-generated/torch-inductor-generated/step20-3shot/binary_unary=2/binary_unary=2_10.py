
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 10, 5, stride=3, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.abs(v1)
        return v2
# Inputs to the model
x1 = torch.randn(10, 3, 224, 224)
