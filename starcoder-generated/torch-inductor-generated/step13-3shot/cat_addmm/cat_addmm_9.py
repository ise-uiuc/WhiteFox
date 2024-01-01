
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2):
        v1 = torch.addmm(x1, x2, torch.rand(6, 10), torch.rand(10, 6))
        v2 = torch.cat([v1], dim=-1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
x2 = torch.randn(6, 10)
