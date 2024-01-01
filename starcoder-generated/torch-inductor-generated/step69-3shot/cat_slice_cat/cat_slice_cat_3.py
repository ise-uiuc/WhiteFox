
class Model(torch.nn.Module):
    def __init__(self, n1, n2):
        super().__init__()
        self.n1 = n1
        self.n2 = n2
 
    def forward(self, x1, x2, x3):
        s1 = [x1, x2, x3]
        v1 = torch.cat(s1, dim=1)
        v2 = v1[:, 0:9223372036854775807]
        v3 = v2[:, 0:self.n2]
        v4 = torch.cat([v1, v3], dim=1)
        return v4

# Initializing the model
m = Model(1, 2)

# Inputs to the model
x1 = torch.randn(1, 32, 64, 64)
x2 = torch.randn(1, 32, 64, 64)
x3 = torch.randn(1, 32, 64, 64)
