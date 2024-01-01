
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv3d(7, 3, 4, stride=2, padding=3)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 - 0.97
        return v2
# Inputs to the model
x = torch.randn(1, 7, 16, 16, 16)
