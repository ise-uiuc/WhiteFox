
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = torch.nn.BatchNorm2d(4)
        self.conv = torch.nn.Conv2d(4, 4, 3)
    def forward(self, x):
        return self.conv(self.norm(x))
# Inputs to the model
x = torch.randn(1, 4, 4, 4)
