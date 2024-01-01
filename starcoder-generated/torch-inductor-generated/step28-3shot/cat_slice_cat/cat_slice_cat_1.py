
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, size):
        x3 = torch.cat(x1, dim=1)
        x4 = x3[:, 0:9223372036854775807]
        x5 = x4[:, 0:size]
        return torch.cat([x3, x5], dim=1)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 8, 64, 64)
size = 4
x3 = torch.cat(x1, dim=1)
