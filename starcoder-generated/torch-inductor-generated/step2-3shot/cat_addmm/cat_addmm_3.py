
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1):
        v1 = torch.addmm(x1, x1, x1)
        v2 = v1.unsqueeze(0)
        v3 = torch.cat([v1, v2, v1])
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(8, 4)
