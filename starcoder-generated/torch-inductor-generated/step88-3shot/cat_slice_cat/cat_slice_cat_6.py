
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x):
        v1 = torch.cat(x, dim=1)
        v2 = v1[:, 0:9223372036854775807]
        v3 = v2[:, 0:64]
        v4 = torch.cat([v1, v3], dim=1)
        return v4

# Initializing the model
model = Model()

# Inputs to the model'
x = [torch.randn(1, x, 92, 72) for x in range(1, 5)]
