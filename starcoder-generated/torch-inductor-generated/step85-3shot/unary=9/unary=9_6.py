
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, kernel_size=1)
    def forward(self, x1):
        v2 = self.conv(x1)
        v3 = v2 + 3
        v4 = v3.clamp(0, 6)
        v5 = v4.div(6)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64) # N/A
