
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(5, 33, 2, stride=3, padding=0)
    def forward(self, x):
        v = self.conv(x)
        return v - 0
# Inputs to the model
x = torch.randn(1, 5, 65, 65)
