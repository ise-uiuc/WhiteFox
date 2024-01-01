
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, v0, v1):
        v2 = torch.cat([v0, v1], dim=1)
        v3 = v2[:,0:9223372036854775807]
        v4 = v3[:,0:9]
        v5 = torch.cat([v2, v4], dim=1)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x0 = torch.randn(2, 9, 20)
x1 = torch.randn(2, 1, 20)
