
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, kernel_size=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.sigmoid(v1)
        v3 = torch.nn.functional.interpolate(v2, scale_factor=4, mode='nearest')
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 112, 112)
