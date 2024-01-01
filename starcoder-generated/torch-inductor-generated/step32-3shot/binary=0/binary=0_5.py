
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(21, 1, 3, stride=2, padding=2)
    def forward(self, x1, other=None, padding1=None):
        v1 = self.conv(x1)
        if other == None:
            other = torch.randn(v1.shape)
        v2 = v1 + other
        v3 = torch.mul(v2, padding1)
        return torch.relu(v3)
# Inputs to the model
x1 = torch.randn(1, 21, 16, 16)
