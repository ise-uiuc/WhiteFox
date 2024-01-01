
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=0)
    def forward(self, x1, x2, padding1=None, padding2=None):
        v1 = self.conv(x1)
        v2 = self.conv2(x2)
        if padding1 == None:
            padding1 = (v1 + v2)
        if padding2 == None:
            padding2 = (v1 + v2)
        return (padding1 + padding2).flatten()
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
