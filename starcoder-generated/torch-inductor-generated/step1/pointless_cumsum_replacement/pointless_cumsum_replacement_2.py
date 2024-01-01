
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
  
    def forward(self, x):
        v1 = torch.randn((3, 3))
        v2 = torch.cumsum(x1, dim=1)
        v3 = torch.cumsum(torch.Tensor(v1), dim=0)
        v4 = torch.cumsum(torch.Tensor(v2), dim=0)
        v5 = torch.cumsum(torch.add(x1,v3), dim=1)
        v6 = torch.add(torch.Tensor(v4),v5)
        return v6

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(3, 3, 3, 3)
