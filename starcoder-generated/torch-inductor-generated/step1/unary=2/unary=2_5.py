
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convt1 = torch.nn.ConvTranspose2d(3, 8, 3, stride=1, padding=1)
        self.convt2 = torch.nn.ConvTranspose2d(4, 6, 3, stride=1, padding=1)
 
    def forward(self, x):
        v1 = self.convt1(x)
        v2 = self.convt2(v1/0.7978845608028654)
        v3 = v1 + torch.tanh(v2/0.044715)
        v4 = v3 * 0.5
        v5 = self.convt2(v4)
        v6 = self.convt1(v2 * v2)
        v7 = v5*torch.tanh(v6) + 0.44715
        v8 = self.convt2(v7)
        v9 = self.convt2(v8)
        v10 = v9 * v9
        v11 = v8 * v10
        v12 = v8 + v11
        return v12

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
