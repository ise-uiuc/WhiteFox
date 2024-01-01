
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 32, 1, stride=1, padding=1, dilation=1)
    def forward(self, x1, padding0=None, padding1=None):
        v1 = self.conv(x1)
        if padding0 == None:
            padding0 = torch.randn(v1.shape)
        return v1
# Inputs to the model
x1 = torch.randn(1, 2, 64, 64)
