
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2 = torch.nn.ConvTranspose2d(8, 3, 3, stride=1, padding=1)
 
    def forward(self, x):
        v1 = self.conv2(x) + 3
        v2 = torch.clamp(torch.clamp(v1, min=0), max=6)
        v3 = v2 / 6
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 8, 64, 64)
