
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 16, 1, stride=1, padding=1)
    def forward(self, x1, other=(torch.ones(1, 16, 64, 64) + 0.01), padding1=None):
        v1 = self.conv(x1)
        if padding1 == None:
            padding1 = torch.randn(v1.shape)
        if other == None:
            other = torch.randn(v1.shape)
        v2 = torch.sigmoid(v1) + other
        return v2
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
