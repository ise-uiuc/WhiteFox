
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_trans = torch.nn.ConvTranspose2d(8, 8, 3, stride=1, padding=1)
 
    def forward(self, x):
        v1 = self.conv_trans(x)
        v2 = v1 + 0.044715
        v3 = v1 * v2
        v4 = self.conv_trans(x)
        v5 = v3 * 0.7978845608028654
        v6 = torch.tanh(v5)
        v7 = v3 + 1
        v8 = v4 * v7
        v9 = v8 + 0.5
        return v9

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 8, 64, 64)
