
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(32, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 8, 3, stride=1, padding=1)
    def forward(self, x1, other=None):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = v1 + v2
        if other == None:
            other = torch.randn(v3.shape)
        v4 = v3 + other
        return v4
# Inputs to the model
x0 = torch.randn(12, 32, 64, 64)
