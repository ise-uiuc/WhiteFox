
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(14, 12, 1, stride=1, padding=1)
    def forward(self, x1, other=None, other1=None, other2=None, other3=None):
        v1 = self.conv(x1)
        if other == None:
            other = torch.randn(v1.shape)
        v2 = v1 + other
        if other1 == None:
            other1 = torch.randn(v1.shape)
        v3 = v2 + other1
        if other2 == None:
            other2 = torch.randn(v1.shape)
        v4 = v3 + other2
        if other3 == None:
            other3 = torch.randn(v1.shape)
        v5 = v4 + other3
        return v5
# Inputs to the model
x1 = torch.randn(1, 14, 64, 64)
