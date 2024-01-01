
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(3, 8, 3, stride=1, padding=1)
 
    def forward(self, x):
        v1 = self.conv_t(x)
        v2 = v1 * 0.5
        v3 = v2 * 0.7978845608028654
        v4 = v1 * 0.044715
        v5 = v4 * v4
        v6 = v5 * v3
        v7 = v2 + v6
        v8 = torch.tanh(v7)
        v9 = v8 + 1
        v10 = v1 * v9
        return v10

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
