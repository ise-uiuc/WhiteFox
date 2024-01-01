
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.deconv = torch.nn.ConvTranspose2d(3, 5, 3, stride=2)
 
    def forward(self, x):
        v1 = self.deconv(x)
        v2 = torch.clamp_min(v1, min_value=0.05)
        return torch.clamp_max(v2, max_value=0.08)

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
