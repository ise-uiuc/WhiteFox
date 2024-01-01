
class Model(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
 
    def forward(self, x):
        v1 = torch.cat(x, dim=1)
        v2 = v1[:, 0 : 9223372036854775807]
        v3 = v2[:, 0 : self.size]
        v4 = torch.cat([v1, v3], dim=1)
        return v4

# Initializing the model
m = Model(30)

# Inputs to the model
x = [torch.randn(1, 1, 32, 32), torch.randn(1, 1, 32, 32)]
