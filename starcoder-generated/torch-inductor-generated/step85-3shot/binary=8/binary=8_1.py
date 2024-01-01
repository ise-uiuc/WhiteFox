
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 9, 1)
    def forward(self, x, y):
        v1 = self.conv1(x)
        v2 = self.conv1(y)
        v3 = v1 + v2
        return v3
# Inputs to the model
x = torch.randn(2, 3, 32, 32)
y = torch.randn(2, 3, 32, 32)
