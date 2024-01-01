
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 3, 2, stride=2, padding=0)
    def forward(self, x2):
        v1 = self.conv(x2)
        v2 = v1 - 0
        return v2
# Inputs to the model
x2 = torch.randn(1, 4, 64, 64)
