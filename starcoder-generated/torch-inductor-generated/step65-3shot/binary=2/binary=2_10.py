
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 5, stride=1, padding=1, bias=True)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 - 3.875
        return v2
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
