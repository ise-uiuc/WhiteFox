
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(5, 32, 10, stride=1, padding=4)
    def forward(self, x1):
        return self.conv(x1)
# Inputs to the model
x1 = torch.randn(1, 5, 64, 64)
