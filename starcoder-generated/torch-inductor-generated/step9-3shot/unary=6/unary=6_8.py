
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(19, 3, 1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.relu6(v1 + 3)
        v3 = v2 / 6
        return v3
# Inputs to the model
x1 = torch.randn(33, 19, 56, 56)
