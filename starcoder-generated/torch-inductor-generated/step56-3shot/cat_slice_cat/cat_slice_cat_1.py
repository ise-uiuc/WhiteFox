
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1):
        v1 = torch.cat(x1, dim=1)
        v2 = v1[:, 0:9223372036854775807]
        v3 = v2[:, 0:list(v2.size())[1]]
        v4 = torch.cat([x1, v3], dim=1)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = [torch.randn(1, 3, 64, 64) for _ in range(3)]
