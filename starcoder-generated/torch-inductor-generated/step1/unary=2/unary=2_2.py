
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.convtrans = torch.nn.ConvTranspose2d(8, 5, 3, stride=1, padding=1)
 
    def forward(self, x):
        v1 = self.convtrans(x)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = torch.tanh(v5)
        v7 = v2 * v6
        v8 = v7 * v7
        v9 = v8 * 0.044715
        v10 = v7 * v9
        v11 = v10 + v1
        return v11

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 8, 64, 64)
