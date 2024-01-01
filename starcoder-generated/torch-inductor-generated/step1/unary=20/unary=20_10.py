
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose3d(1, 8, 3, bias=False)
 
    def forward(self, x):
        v1 = self.conv(x)
        v2 = torch.sigmoid(v1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 1, 64, 64)
