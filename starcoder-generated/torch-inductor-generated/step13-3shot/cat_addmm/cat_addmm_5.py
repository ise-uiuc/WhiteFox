
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.randn(3, 3))
 
    def forward(self, x2):
        v1 = torch.cat([x2], dim=1)
        v2 = torch.addmm(input, self.weights, v1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(1, 512, 3, 3)
