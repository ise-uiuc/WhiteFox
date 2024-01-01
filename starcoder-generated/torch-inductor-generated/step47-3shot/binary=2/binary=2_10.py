
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x, x2=torch.randn(2, 3, 128, 128)):
        v1 = self.conv(x, padding=1, bias=None)
        v2 = v1 - x2
        return v2
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
