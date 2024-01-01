
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2):
        v1 = torch.cat([x1, x2], dim=1)
        v2 = v1[:, 0:x1.shape[1]+x2.shape[1]-1]
        v3 = v2[:, 0:v2.shape[1]-13]
        v4 = torch.cat([v1, v3], dim=1)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 2, 71, 71)
x2 = torch.randn(1, 5, 23, 23)
