
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 1, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        return v1.data
# Inputs to the model
x1 = torch.randn(5, 3, 64, 64)
