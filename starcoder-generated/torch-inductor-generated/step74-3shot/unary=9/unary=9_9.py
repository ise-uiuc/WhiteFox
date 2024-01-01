
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        # Clamp operations
        v3 = torch.clamp_min(x1, 0.0)
        v4 = torch.clamp_max(v3, 6.0)

        # Convolut operations
        v1 = self.conv(v4)

        # Divide operation
        v5 = v1.div(6.0)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
