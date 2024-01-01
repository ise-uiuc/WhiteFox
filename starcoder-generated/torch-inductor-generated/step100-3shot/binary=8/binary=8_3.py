
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 4, 5, stride=1, padding=2)
    def forward(self, x1, x2, x3):
        v1 = self.conv1(x1)
        v2 = v1 + x2
        v3 = v1 + x3
        return v2, v3
# Inputs to the model
x1 = torch.randn(1, 1, 56, 56)
x2 = torch.randn(1, 1, 56, 56)
x3 = torch.randn(1, 1, 56, 56)
