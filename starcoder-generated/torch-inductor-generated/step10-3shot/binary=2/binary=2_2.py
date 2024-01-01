
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(6, 6, 1, stride=1, padding=0)
    def forward(self, x):
        v = self.conv(x)
        v2 = v - False
        return v2
# Inputs to the model
x1 = torch.randn(1, 6, 224, 224)
