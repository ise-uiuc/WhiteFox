
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convA = torch.nn.Conv2d(2, 2, 2, stride=2, padding=2)
        self.convB = torch.nn.Conv2d(2, 2, 3, stride=3, padding=3)
    def forward(self, x1, other):
        v1 = self.convA(x1)
        v2 = self.convB(v1)
        v3 = v2 * other
        v4 = v3 * other
        return v1
# Inputs to the model
x1 = torch.randn(1, 2, 16, 16)
other = torch.randn(1, 2, 4, 4)
