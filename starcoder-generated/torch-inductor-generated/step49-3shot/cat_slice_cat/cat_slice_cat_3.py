
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3):
        v1 = torch.cat([x1, x2, x3], 1)
        v2 = v1[:, ::5]
        v3 = v2[:, :x1.shape[1]]
        v4 = torch.cat([x1, v3], 1)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10, 64, 64)
x2 = torch.randn(1, 20, 128, 128)
x3 = torch.randn(1, 50, 256, 256)
