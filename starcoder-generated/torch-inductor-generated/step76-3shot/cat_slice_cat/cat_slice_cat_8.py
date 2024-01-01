
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cat1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.cat2 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=1)
 
    def forward(self, x1, x2):
        v1 = torch.cat([x1, x2], dim=1)
        v2 = v1[:, 0:9223372036854775807]
        v3 = v2[:, :16]
        v4 = torch.cat([v1, v3], dim=1)
        v5 = self.cat1(v4) + self.cat2(v3)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
