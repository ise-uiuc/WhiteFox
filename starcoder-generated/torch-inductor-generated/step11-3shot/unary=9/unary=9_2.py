
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 + 3 # Add 3 to the output of the convolution
        v3 = torch.clamp_min(v2,0)
        v4 = torch.clamp_max(v3,6) # Clamp the output to a minimum of 0 and a maximum of 6
        v5 = torch.div(v4,6)
        return v5 / 6
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
