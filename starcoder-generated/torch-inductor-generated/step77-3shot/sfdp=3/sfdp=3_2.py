
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        pass
 
    def forward(self, input, p, dim=1):
        pass

# Initializing the model
m = Model()

# Inputs to the model
input = torch.randn(4, 5, 3, 3)
p = 0.4
dim = 1
