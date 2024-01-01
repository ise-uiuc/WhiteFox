
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 2, 1, stride=1, padding=0)
    def forward(self, x1):
        return self.conv(x1)
# Inputs to the model
x1 = torch.randn(1, 1, 1, 1)
