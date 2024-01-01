
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convt = torch.nn.ConvTranspose2d(16, 8, 3, stride=1, padding=1)
 
    def forward(self, x):
        v1 = self.convt(x)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        return v6

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(2, 16, 32, 32)
