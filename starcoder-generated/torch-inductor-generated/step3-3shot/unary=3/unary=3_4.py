
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=3, padding=1)
    def forward(self, x1):
        v = self.conv1(x1)
        return v
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
