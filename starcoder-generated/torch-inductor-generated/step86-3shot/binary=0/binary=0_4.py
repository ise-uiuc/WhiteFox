
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x, padding1=None):
        v1 = self.conv(x) + torch.randn(v1.shape)
        if padding1 == None:
            padding1 = torch.randn(v1.shape)
        v2 = v1 + padding1
        return v1
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
