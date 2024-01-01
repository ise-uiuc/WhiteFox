
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2):
        v1 = torch.cat([x1, x2], dim=1)
        v2 = v1[:, 0:v1.shape[1] - 1]
        v3 = v2[:, 0:x1.shape[2]]
        v4 = torch.cat([v1, v3], dim=1)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 4, 8, 8)
x2 = torch.randn(1, 3, 8, 8)
