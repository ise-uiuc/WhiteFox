
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3, stride=1, padding=1)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 - torch.randn(1, 3, 3, 3)
        return v2
# Inputs to the model
x = torch.randn(1, 3, 5, 5)
