
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.deconv = torch.nn.ConvTranspose2d(8, 3, 3, stride=1)
 
    def forward(self, x):
        v1 = self.deconv(x)
        v2 = self.deconv(x)
        v3 = v1 + 0
        v4 = v3 * 0.044715 + (0.7978845608028654 + v2)
        v5 = torch.tanh(v4)
        v6 = v5 + 1 - 1
        v7 = v6 * 0.5 + (v1 * 0.5)
        return v7

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 8, 64, 64)
