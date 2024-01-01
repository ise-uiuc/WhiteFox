
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp(v1, self.minimum, self.maximum)
        return v2
# Inputs to the model
x1 = torch.randn(4, 3, 64, 64)
model = Model()
model.minimum = 0
model.maximum = 6
