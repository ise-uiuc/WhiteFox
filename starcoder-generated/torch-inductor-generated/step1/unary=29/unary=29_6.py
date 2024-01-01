
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(3, 8, 3, stride=1, padding=1)
 
    def forward(self, x, min_value=0, max_value=1):
        v1 = self.conv(x)
        v2 = torch.clamp_min(min_value, v1)
        v3 = torch.clamp_max(max_value, v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
