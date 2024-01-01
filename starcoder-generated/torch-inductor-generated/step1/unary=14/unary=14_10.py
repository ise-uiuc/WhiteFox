
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tconv = torch.nn.ConvTranspose2d(1, 8, 3, stride=1, padding=1)
 
    def forward(self, x):
        v1 = self.tconv(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 1, 64, 64)
