
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(4, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(4, 8, 1, stride=1, padding=1)
    def forward(self, x1, x2):
        v1 = self.conv1(x1) + self.conv1(x2)
        v2 = v1 - 2.0
        return v2
# Inputs to the model
x1 = torch.randn(1, 4, 65, 65)
x2 = torch.randn(1, 4, 64, 64)
