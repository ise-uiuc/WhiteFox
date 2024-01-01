
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2):
        v1 = [x1, x2]
        v2 = torch.cat(v1, dim=1)
        v3 = v2[:, 0:9223372036854775807]
        v4 = v3[:, 0:64]
        v5 = torch.cat([v2, v4], dim=1)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64, device='cpu')
x2 = torch.randn(1, 3, 128, 128, device='cpu')
