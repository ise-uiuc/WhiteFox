
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight_1 = torch.nn.Parameter(torch.rand(2, 3))
        self.weight_2 = torch.nn.Parameter(torch.rand(2, 2))
 
    def forward(self, x):
        v1 = torch.einsum('ij,jk->ik', x, self.weight_1)
        v2 = torch.full_like(v1, 1)
        v3 = v1.type(torch.float32)
        v4 = v2.type(torch.float32)
        v5 = v3 + v4
        v6 = v2 + v3
        v7 = v4 + v5
        v8 = torch.cumsum(v7, 0)
        return v8

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(2, 2)

# Computing output of the model
