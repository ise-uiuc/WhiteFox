
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 32, 1, stride=1, padding=1)
    def forward(self, x1, other=0):
        v1 = self.conv(x1)

        other = other.reshape(v1.shape)
        v2 = v1 + other
        return v2
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
