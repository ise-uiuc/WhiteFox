
class Model(torch.nn.Module):
    def __init__(self, in_size):
        super().__init__()
 
    def forward(self, x1, x2):
        v1 = torch.cat([x1, x2], dim=1)
        v2 = v1[:, 1:9223372036854775807]
        v3 = v2[:, 1:in_size]
        v4 = torch.cat([v1, v3], dim=1)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 14, 64, 64)
x2 = torch.randn(1, 13, 64, 64)
in_size = x1.shape[1]
