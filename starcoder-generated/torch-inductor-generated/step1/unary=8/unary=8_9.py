
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_tr = torch.nn.ConvTranspose2d(3, 8, 3, stride=1, padding=1)
 
    def forward(self, x):
        v1 = self.conv_tr(x)
        v2 = v1 + 3
        v3 = torch.clamp(v2, 0, 6)
        v4 = v2 * v3
        return v4 / 6

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 1, 2, 2)
