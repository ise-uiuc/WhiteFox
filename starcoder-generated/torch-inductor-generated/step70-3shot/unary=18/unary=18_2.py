
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 64, 11, stride=1, padding=5, bias=False)
        self.conv1 = torch.nn.Conv2d(3, 64, 11, stride=1, padding=5, bias=True)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv1(x1)
        return v1, v2
# Inputs to the model
x1 = torch.randn(1, 3, 300, 300)
