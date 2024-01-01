
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.addc = torch.nn.Conv2d(3, 8, 1, stride=3)
    def forward(self, x1, x2):
        v1 = self.addc(x1)
        v3 = torch.relu(v1)
        v4 = self.addc(x2)
        v6 = torch.relu(v4)
        v5 = v3 + v6
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
