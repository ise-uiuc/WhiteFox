
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l = torch.nn.Linear(10, 20)
 
    def forward(self, x1, other):
        v1 = self.l(x1)
        v2 = v1 + other
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 10)
other = torch.randn(2, 20)
