
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(768, 1024, 1, stride=1, padding=0, bias=True)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = 2 + v1
        v3 = v2.clamp(min=0, max=6)
        v4 = v3.div(6)
        return v4
# Inputs to the model
x1 = torch.randn(16, 768, 4, 4)
