
class Model(torch.nn.Module):
    def __init__(self, a=1):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 9, stride=4, padding=4)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 - 3.5
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
