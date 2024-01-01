
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(32, 32, 2, stride=2, padding=1)
    def forward(self, x):
        return self.conv(x)
# Inputs to the model
x = torch.rand(1, 32, 100, 100)
