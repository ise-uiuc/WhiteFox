
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1):
        v1 = x1 / 512.0
        v2 = x1 / 16384.0
        v3 = x1 / 524288.0
        v4 = x1 / 1.048576e+06
        v5 = x1 / 5.149056e+06
        v6 = torch.concat((v1,v2,v3,v4,v5), 1)
        return v6

# Initializing the model
m = Model()

# Getting the input shape of the model
x1 = torch.randn(1, 3, 64, 64)
print(m(x1).shape)

