
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(8, 8, 1, stride=1,padding=0)
    def forward(self, x1, x2, other=1, padding1=None, padding2=None):
        v1 = self.conv1(x1)
        if padding1 == None:
            padding1 = torch.randn(v1.shape)
        v2 = self.conv2(x2, padding1)
        if padding2 == None:
            padding2 = torch.randn(v2.shape)
        v3 = v1 + other
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
