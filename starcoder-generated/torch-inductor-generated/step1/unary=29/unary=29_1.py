
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2 = torch.nn.ConvTranspose2d(4, 4, 3, stride=1, padding=1)
 
    def forward(self, x):
        v1 = self.conv2(x)
        v2 = v1.clamp_min(1.1)
        v3 = v2.clamp(min=0)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 4, 6, 6)
