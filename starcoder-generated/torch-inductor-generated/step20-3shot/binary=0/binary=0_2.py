
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(5, 7, 1, stride=1, padding=1)
    def forward(self, x1, other, padding1=None, conv=None):
        v1 = self.conv(x1)
        if conv == None:
            conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        v2 = v1 + other
        return v2
# Inputs to the model
x1 = torch.randn(1, 5, 32, 32)
other = torch.randn(10)
