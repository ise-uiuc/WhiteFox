
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(32, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=1)
    def forward(self, x1, other1=1, other2=2, padding1=None, padding2=None):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = v2 + other1
        v4 = v2 + other2
        if padding1 == None:
            padding1 = torch.randn(v3.shape)
        if padding2 == None:
            padding2 = torch.randn(v4.shape)
        v5 = v3 + padding1
        v6 = v4 + padding2
        return v6
# Inputs to the model
x1 = torch.randn(1, 32, 64, 64)
