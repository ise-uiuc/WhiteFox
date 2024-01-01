
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convT = torch.nn.ConvTranspose2d(1, 1, 4, stride=1, padding=1)
 
    def forward(self, x):
        v1 = self.convT(x)
        v2 = v1 > 0
        v3 = torch.where(v2, v1, v1*(-0.1))
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 1, 16, 16)
