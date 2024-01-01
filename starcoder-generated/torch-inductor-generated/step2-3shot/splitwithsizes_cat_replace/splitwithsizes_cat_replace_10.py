
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1):
        v1 = torch.split(x1, [2 for i in range(x1.size(1))], dim=1)
        v2 = torch.cat([v1[0], v1[1], v1[2]], dim=1)
        return len(v2) > 0

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 14, 64, 64)
