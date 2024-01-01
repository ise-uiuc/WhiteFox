
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1):
        v2 = x1.view(x1.size(0), -1)
        v3 = torch.linalg.norm(v2)
        v1 = v2 / v3
        return v1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(5, 10)
