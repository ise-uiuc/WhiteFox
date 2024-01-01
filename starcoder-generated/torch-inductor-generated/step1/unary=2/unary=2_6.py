
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(3, 8, 3, stride=1, padding=1)
 
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 * 0.5
        v3 = v1 + 0.5
        v4 = v1 * v2
        v5 = v4 * v2
        v6 = v3 * 0.044715
        v7 = v3 + v6
        v8 = torch.tanh(v7)
        v9 = v2 * 0.7978845608028654
        v10 = v2 + v9
        v11 = v1 * v10
        v12 = v11 * v10
        return v12

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
