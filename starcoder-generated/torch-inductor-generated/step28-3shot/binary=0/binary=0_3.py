
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(5, 8, 2, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(5, 8, 2, stride=1, padding=1)
    def forward(self, x1, x2, v3, padding1=None, padding2=None, v6=None):
        v1 = self.conv1(x1)
        v2 = self.conv2(x2)
        if padding1 is None:
            padding1 = torch.randn(v1.shape)
        if padding2 is None:
            padding2 = torch.randn(v2.shape)
        if v6 is None:
            v6 = torch.randn(v1.shape)
        v4 = v1 + padding1
        v5 = v4 + v6
        v7 = v5 + v3
        return v7
# Inputs to the model
x1 = torch.randn(3, 5, 1, 1)
v2 = torch.randn(3, 5, 1, 1)
v3 = torch.randn(3, 8, 1, 1)
