
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.slice = random.randint(1, 8)
 
    def forward(self, x1, x2, x3):
        v1 = torch.cat([x1, x2, x3], dim=1)
        v2 = v1[:, :self.slice]
        v3 = v2[:, :8]
        v4 = torch.cat([v1, v3], dim=1)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, random.randint(1, 8), random.randint(1, 8), random.randint(1, 8))
x2 = torch.randn(1, 3, random.randint(1, 8), random.randint(1, 8), random.randint(1, 8))
x3 = torch.randn(1, 3, random.randint(1, 8), random.randint(1, 8), random.randint(1, 8))
