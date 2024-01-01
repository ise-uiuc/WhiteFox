
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(34, 4, 1, stride=1, padding=1)
    def forward(self, x1, padding1=None):
        v1 = self.conv(x1)
        if padding1 == None:
            padding1 = torch.randn(v1.shape)
        return v1
# Inputs to the model
x1 = torch.randn(1, 34, 68, 68)
