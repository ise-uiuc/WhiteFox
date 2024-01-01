
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = torch.nn.Conv2d(1, 8, 1, stride=2, padding=0)
    def forward(self, x1):
        v1 = self.c1(x1)
        v2 = self.c1(x1)
        v3 = self.c1(x1)
        v4 = x1 + v1 + v2 + v3
        v5 = torch.relu(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
