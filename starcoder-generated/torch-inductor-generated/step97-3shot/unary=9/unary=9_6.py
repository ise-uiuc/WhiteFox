
class Model(torch.nn.Module):
    def __init__(self):
       super().__init__()
       self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
       self.weight = torch.zeros(self.conv.weight.shape) + 3
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 + self.weight
        t1 = torch.clamp(v2, min=0, max=6)
        v3 = torch.div(t1, 6)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
