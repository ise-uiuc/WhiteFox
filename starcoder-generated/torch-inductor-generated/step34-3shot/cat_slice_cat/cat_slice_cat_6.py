
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, a1, a2):
        v1 = torch.cat([a1, a2], dim=1)
        v2 = v1[:, :, :, 0:9223372036854775807]
        v3 = v2[:, :, :, 0:10]
        v4 = torch.cat([v1, v3], dim=1)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
a1 = torch.randn(1, 3, 60, 10)
a2 = torch.randn(1, 3, 50, 10)
