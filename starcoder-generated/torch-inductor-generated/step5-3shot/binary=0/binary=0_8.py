
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1, other=np.random.randn(1, 3, 64, 64)):
        v1 = self.conv(x1)
        if np.array_equal(other.shape, v1.shape):
            other = 0
        v2 = v1 + other
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
