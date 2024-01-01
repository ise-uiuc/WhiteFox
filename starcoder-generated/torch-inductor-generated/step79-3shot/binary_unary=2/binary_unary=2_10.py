
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(4, 16, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.abs(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 4, 10, 10)
