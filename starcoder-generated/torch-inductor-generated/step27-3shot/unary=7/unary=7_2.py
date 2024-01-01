
class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 3, 1, stride=1, padding=1)
 
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 * torch.clamp(torch.clamp(v1 + 3, 0, 7) - 3, -7, 0)
        v3 = v2 / 6
        return v3

# Initializing the model
m = Model1()

# Inputs to the model
x1 = torch.randn(1, 2, 64, 64)
