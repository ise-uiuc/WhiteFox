
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 24, 8, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(24, 48, 6, stride=1, padding=0)
    def forward(self, x1, other=None, padding1=None):
        v1 = self.conv1(x1) + x1
        if other == None:
            other = torch.randn(v1.shape)
        if padding1 == None:
            padding1 = torch.randn(v1.shape)
        v2 = v1 + other
        v3 = self.conv2(v2 - padding1)
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 224, 244)
