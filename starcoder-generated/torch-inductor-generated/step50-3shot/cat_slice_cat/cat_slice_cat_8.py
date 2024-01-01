
class Model(torch.nn.Module):
    def __init__(self):
        pass
 
    def forward(self, x1):
        x2 = x1.flatten().reshape(-1, 400000000)
        x3 = x2[:, 0:0x80000000]
        x4 = x2[:, 0:0x140000000]
        x5 = torch.concatenate([x2, x4], dim=1)
        return x5

# Initializing the model
m = Model()

# Inputs to the model
shape = [0, 4, 1, 64, 64]
dtype = torch.int
x1 = torch.randint(0, 0x140000000, shape, dtype, True)
