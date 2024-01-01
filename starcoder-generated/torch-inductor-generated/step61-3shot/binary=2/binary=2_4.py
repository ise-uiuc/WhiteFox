
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 5, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 - torch.tensor(1e+295)
        return v2
# Inputs to the model
x1 = torch.randn(16, 5, 5, 5)
