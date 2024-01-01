
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x):
        v1 = torch.cat(x, dim=1)
        v2 = v1[:, :-1]
        v3 = v1[:, :max_size]
        v4 = torch.cat([v1, v3], dim=1)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10, 23)
x2 = torch.randn(1, 10, 23)
x3 = torch.randn(1, 10, 23)
