
class Model(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
 
    def forward(self, *x):
        v1 = x[0]
        for t in x[1:]:
            v1 = torch.cat([v1, t], dim=1)
        v2 = torch.squeeze(v1)
        v3 = v2[:, 0:self.size]
        v4 = torch.cat([v1, v3], dim=1)
        return v4

# Initializing the model
m = Model(10)

# Inputs to the model
x1 = torch.randn(1, 2, 10)
x2 = torch.randn(1, 1, 10)
