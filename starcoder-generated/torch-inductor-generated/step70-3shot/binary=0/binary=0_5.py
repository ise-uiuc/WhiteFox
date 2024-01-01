
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(192, 64, 3, stride=1, padding=1)
    def forward(self, x1, x2, padding1=None):
        v1 = self.conv(x1)
        if padding1 == None:
            padding1 = torch.randn(v1.shape)
        v2 = padding1 + x1
        v3 = v2 + x2
        v4 = torch.relu(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 192, 64, 64)
x2 = torch.randn(1, 192, 64, 64)
