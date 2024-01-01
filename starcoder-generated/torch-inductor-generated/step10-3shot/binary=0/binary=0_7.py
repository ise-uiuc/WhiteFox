
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 8, 1, stride=1, padding=1)
    def forward(self, x1, x2, padding1=None, x3=None):
        v1 = self.conv1(x1)
        v2 = self.conv2(x2)
        if x3 == None:
            x3 = torch.randn(v1.shape)
        v3 = v1 + x3
        if padding1 == None:
            padding1 = torch.randn(v3.shape)
        v4 = v3 + v2
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 32, 64, 64)
