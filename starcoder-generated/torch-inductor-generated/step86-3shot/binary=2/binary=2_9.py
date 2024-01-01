
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 5, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = True - self.conv(x1)
        v2 = False - v1
        return v2
# Inputs to the model
x1 = torch.randn(1, 1, 100, 100)
