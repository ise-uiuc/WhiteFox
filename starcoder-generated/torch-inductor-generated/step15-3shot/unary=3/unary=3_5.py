
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 32, 8, stride=2, padding=7)
    def forward(self, x1):
        return self.conv(x1)
# Inputs to the model
x1 = torch.randn(1, 1, 512, 512)
