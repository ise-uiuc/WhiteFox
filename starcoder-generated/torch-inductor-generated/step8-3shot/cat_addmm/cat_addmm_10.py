
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2):
        v1 = torch.addmm(x1, x2, x2)
        return [v1, x2, x1]

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(10, 10)
x2 = torch.randn(10, 10)
