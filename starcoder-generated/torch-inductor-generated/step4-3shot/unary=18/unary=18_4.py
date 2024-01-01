
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 18, kernel_size=3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v3 = torch.sigmoid(v1)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
