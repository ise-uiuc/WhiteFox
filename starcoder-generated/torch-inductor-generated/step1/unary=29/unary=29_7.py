
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(3, 8, 3, stride=1, padding=1, output_padding=1)
 
    def forward(self, x):
        v1 = self.conv_t(x)
        v2 = v1.clamp_min(-0.5)
        v3 = v2.clamp_max(0.6)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
