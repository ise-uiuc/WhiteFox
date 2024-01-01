
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(26, 26, 4, 2, 0)
    def forward(self, x):
        x = self.conv(x)
        x = self.conv(x)
        return x
# Inputs to the model
x = torch.randn(1, 26, 5, 5)
