
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 30, 30)
    def forward(self, x3):
        v1 = self.conv(x3)
        v2 = v1 - -124.1
        return v2
# Inputs to the model
x3 = torch.randn(2, 3, 2, 2, 2)
