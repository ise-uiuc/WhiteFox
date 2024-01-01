
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1):
        v3 = torch.cat(x1, dim=1)
        v4 = v3[:, 0:9223372036854775807]
        v5 = v4[:, 0:32]
        return torch.cat([v3, v5], dim=1)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 16, 5, 5)
