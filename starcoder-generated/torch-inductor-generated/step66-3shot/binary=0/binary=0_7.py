
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=0)
    def forward(self, x1, other=1, conv2d=None):
        if conv2d == None:
            conv2d = torch.nn.Conv2d(other, 8, 1, stride=1, padding=0)
        v1 = self.conv(x1)
        v2 = v1 + other
        return v2
# Inputs to the model
x1 = torch.randn(3, 3, 10, 10)
