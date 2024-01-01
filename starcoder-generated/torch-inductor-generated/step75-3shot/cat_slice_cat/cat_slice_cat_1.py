
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
 
    def forward(self, x1, x2, x3):
        v1 = torch.cat([x1, x2, x3], dim=1)
        v3 = v1[:, 0:9223372036854775807]
        v5 = v3[:, 0:min(v1.shape[2:])]
        v6 = torch.cat([v1, v5], dim=1)
        return v6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 3, 64, 32)
x2 = torch.randn(2, 3, 64, 24)
x3 = torch.randn(2, 3, 64)
