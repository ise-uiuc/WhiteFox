
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 32, 2, stride=2, padding=0, groups=1, bias=True)
    def forward(self, x1):
        v1 = self.conv(x1)
        return torch.sigmoid(v1)
# Inputs to the model
x1 = torch.randn(2, 16, 23, 23)
