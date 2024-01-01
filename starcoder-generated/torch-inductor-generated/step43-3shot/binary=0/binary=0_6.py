
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv3d(1, 16, 3, stride=3, padding=1)
    def forward(self,x1, other=False):
        v1 = self.conv(x1)
        v2 = v1 + other
        return v2
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64, 64)
