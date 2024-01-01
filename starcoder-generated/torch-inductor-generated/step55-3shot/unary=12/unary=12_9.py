
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1.sigmoid()
        v2 = 2 * v2 # Adding 2
        v3 = v2.mul(2 * v1) # Multiplying 2
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
