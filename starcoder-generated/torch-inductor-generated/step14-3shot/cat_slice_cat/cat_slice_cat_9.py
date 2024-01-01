
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x):
        v1 = torch.cat([x, x+1, x+2, x+3, x+4], dim=1)
        v2 = v1[:, 0:9223372036854775807]
        v3 = v2[:, 0:int(64/x.size(0))]
        v4 = torch.cat([v1, v3], dim=1)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(16, 3, 64, 64)
