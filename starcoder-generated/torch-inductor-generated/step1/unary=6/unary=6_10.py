
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
 
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 + 3
        v3 = np.clip(v2, 0, 6)
        v4 = v1 * v3
        v5 = 6.
        v6 = v4 / v5
        return v6

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
