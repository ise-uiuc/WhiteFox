
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1):
        v1 = torch.split(x1, (2,8,8,2), dim=1)
        v2 = torch.cat([v1[i] for i in range(len(v1))], dim=2)
        v3 = torch.split(v2, (2,4,2), dim=1)
        v4 = torch.cat([v3[i] for i in range(len(v3))], dim=1)
        return True

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 32, 14, 14)
