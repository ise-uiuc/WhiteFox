
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 3, 1, stride=1, padding=1)
    def forward(self, x, s=2):
        v1 = self.conv(x)
        v2 = torch.sub(v1, s)
        return v2
# Inputs to the model
x = torch.randn(1, 16, 128, 128)
