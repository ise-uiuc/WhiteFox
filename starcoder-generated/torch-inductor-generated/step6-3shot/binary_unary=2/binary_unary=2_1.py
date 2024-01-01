
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv3d(3, 8, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 - torch.rand(1, 1, 1, 1, 1)
        v3 = nn.ReLU()(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64, 64)
