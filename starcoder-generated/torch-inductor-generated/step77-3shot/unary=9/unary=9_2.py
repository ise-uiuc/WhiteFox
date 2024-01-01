
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        t1 = self.conv(x1)
        v1 = t1 + 3
        v2 = v1.clamp(min=0, max=6)
        t2 = v2.div(6)
        output = t2
        return output
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
