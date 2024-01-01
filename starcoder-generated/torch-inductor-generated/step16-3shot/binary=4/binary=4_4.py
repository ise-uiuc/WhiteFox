
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l = torch.nn.Linear(25, 25)
 
    def forward(self, x1, x2):
        v1 = self.l(x1)
        v2 = v1 + x2
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 25)
x2 = torch.randn(1, 25)
