
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(3, 8, 3, stride=1, padding=1)
 
    def forward(self, x):
        v1 = x
        v2 = torch.clamp_min(v1, min_value=0)
        v3 = torch.clamp_max(v2, max_value=10)
        v4 = self.conv(v3)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
