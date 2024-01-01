
class Model(torch.nn.Module):
    def __init__(self, s):
        super().__init__()
        self.s = s
 
    def forward(self, x1, x2):
        v1 = torch.cat([x1, x2], dim=1)
        v2 = v1[:, 0:9223372036854775807]
        v3 = v2[:, :self.s]
        v4 = torch.cat([x1, v3], dim=1)
        return v4

# Initializing the model
m = Model(2)

# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
x2 = torch.randn(1, 2, 64, 64)
