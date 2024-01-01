
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convt = torch.nn.ConvTranspose2d(3, 8, 3, stride=1, padding=1, output_padding=0)
 
    def forward(self, x):
        v0 = self.convt(x)
        v1 = v0 * 3
        v2 = v1 + 0
        v3 = v2.clamp_min(0)
        v4 = v3.clamp_max(6)
        v5 = v0 * v4
        v6 = v5 / 6
        return v6

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3, 48, 48)
