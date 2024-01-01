
class Model(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
 
    def forward(self, x1):
        v4 = torch.cat([x1, x1], dim=1)
        v3 = v4[:, 0:size]
        v2 = v3[:, 0:size]
        v1 = torch.cat([v4, v2], dim=1)
        return v1

# Initializing the model
m = Model(128)

# Inputs to the model
x1 = torch.randn(1, 2048, 1, 1)
