
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(8, 10, 1, stride=1, padding=1)
    def forward(self, x1, other=None, other2=None):
        v1 = self.conv(x1)
        if other == None:
            other = torch.randn(v1.shape)
        v2 = v1 + other
        if other2 == None:
            other2 = torch.randn(v1.shape)
        v3 = v2 + other2
        return v3
# Inputs to the model
x1 = torch.randn(1, 8, 64, 64)
