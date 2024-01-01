
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x):
        v1 = torch.stack([0.125, 0.125])
        v2 = torch.stack([0.25, 0.25])
        v3 = torch.cat([v1, v2])
        v4 = v3[:, 0:1]
        v5 = v3[:, 0:x.size(2)]
        v6 = torch.cat([v3, v5])
        return v6

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(4, 1, 5000)
