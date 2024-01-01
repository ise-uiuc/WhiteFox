
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1):
        l1 = [x1, x1, x1, x1]
        v1,v2,v3,v4 = torch.split(x1, 1, 3)
        l1[0] = v1
        l1[1] = v2
        l1[2] = v3
        l1[3] = v4
        v5 = l1[2] + v3
        v6 = torch.cat([v5, v3, v4, v3], 3)
        return v6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8, 64, 64)
