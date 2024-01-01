
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3):
        x4 = torch.cat((x1, x2, x3), 1)
        v1 = x4[:, -1]
        v2 = torch.flip(x1, [2, 3])
        v3 = v2[:, ::2, ::2]
        v4 = torch.cat([x4, v3], 1)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
x2 = torch.randn(1)
x3 = torch.randn(1)

# 