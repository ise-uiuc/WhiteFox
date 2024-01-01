
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(8, 3, 1, stride=1, padding=1)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = v1 - 0.1782
        return v2
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
