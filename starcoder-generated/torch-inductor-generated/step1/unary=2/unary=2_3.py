
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.deconv = torch.nn.ConvTranspose2d(1, 4, 3, stride=1, padding=1)
 
    def forward(self, x):
        v1 = x + 0.044715
        v2 = v1 * 0.7978845608028654
        v3 = v2.tanh()
        v4 = self.deconv(v3)
        v5 = v4 * 0.5
        v6 = v4 * v5
        return v6

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 1, 32, 32)
