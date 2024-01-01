
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1, groups=3)
    def forward(self, x1):
        v1 = self.conv1(x1)
        return v1
# Inputs to the model
x = torch.randn(1, 3, 32, 32)
