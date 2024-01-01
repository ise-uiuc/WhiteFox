
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, 3, stride=2, padding=1)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 - torch.randn(16, 3, 3, 3)
        v3 = torch.max(v2)
        return v3
# Inputs to the model
x = torch.randn(1, 3, 128, 128)
