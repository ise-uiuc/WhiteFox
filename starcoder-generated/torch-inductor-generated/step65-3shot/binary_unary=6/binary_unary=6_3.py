
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)
 
    def forward(self, x0):
        v1 = self.linear(x0)
        v2 = v1 - 1
        v3 = v2.exp()
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x0 = torch.randn(1, 4)
