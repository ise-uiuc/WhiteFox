
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 1)
    def forward(self, x):
        return self.conv(x)
# Inputs to the model
x = torch.randn(1, 10, 11, 11)
