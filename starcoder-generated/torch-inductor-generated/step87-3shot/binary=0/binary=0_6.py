
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(62, 9, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(62, 2, 1, stride=1, padding=0)
    def forward(self, x1, other=None, padding1=None, other2=None, padding2=None):
        v1 = self.conv(x1)
        v2 = self.conv2(x1)
        v3 = v1 + v2
        if other == None:
            other = torch.randn(v3.shape)
        v4 = v3 + other
        return v4
# Inputs to the model
x1 = torch.randn(1, 62, 32, 32)
