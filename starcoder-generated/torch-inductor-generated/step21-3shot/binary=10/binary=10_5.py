 Definition
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)

    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.nn.functional.interpolate(v1, scale_factor=2)
        v3 = v2 * 0.5
        v4 = torch.nn.functional.relu(v1)
        v5 = torch.nn.functional.interpolate(v4, scale_factor=4)
        v6 = v3 + v5
        return v6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 128, 128)
