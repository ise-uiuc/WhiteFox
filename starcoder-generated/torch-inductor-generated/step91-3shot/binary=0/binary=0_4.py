
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(66, 92, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(15, 34, 1, stride=1, padding=0)
    def forward(self, x1, other=None, padding1=None):
        v1 = self.conv1(x1)
        v2 = self.conv2(x1)
        v3 = v1 + v2
        if other == None:
            other = torch.randn(v1.shape)
        v4 = v3 + other
        return v4
# Inputs to the model
x1 = torch.randn(1, 66, 64, 64)
