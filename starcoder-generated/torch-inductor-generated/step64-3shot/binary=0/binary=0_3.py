
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 14, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(14, 4, 1, stride=1, padding=1)
    def forward(self, x, padding1=None, padding2=None):
        v1 = self.conv(x)
        if padding1 == None:
            padding1 = torch.randn(v1.shape)
        v3 = self.conv2(v1)
        if padding2 == None:
            padding2 = torch.randn(v3.shape)
        v4 = v3 + padding1
        return v4 + padding2
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
