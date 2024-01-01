
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
    def forward(self, x1, x2, x3, x4):
        y = torch.cat([x1, x2, x3, x4], dim=-1)
        return y

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3,  64,  64)
x2 = torch.randn(1, 32,  64,  64)
x3 = torch.randn(1, 64, 128, 128)
x4 = torch.randn(1, 64, 32,  32)
print(m(x1, x2, x3, x4).shape)
