
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
 
    def forward(self, x):       
        v1 = self.conv(x) + 3
        v2 = torch.clamp(v1, min=0, max=6)
        return v2 * 6

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
