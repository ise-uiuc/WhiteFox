
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 8, (3, 3))
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1
        v3 = v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
