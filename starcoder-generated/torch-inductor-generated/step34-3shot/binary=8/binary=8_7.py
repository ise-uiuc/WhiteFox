
class Model(torch.nn.Module):
    def __init__(self, other):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
            self.other = torch.ones_like(other)
 
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 + self.other
        return v2

# Inputs to the model (with random `other` tensor)
x1 = torch.randn(1, 3, 64, 64)
other = torch.randn(8, 3, 1, 1)
a = a[:, None, :].expand(x1.size(0), -1, -1, -1)
