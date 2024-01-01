
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Linear(3, 5)
        self.b = torch.nn.Parameter(torch.zeros(5))
 
    def forward(self, x1):
        v1 = self.w(x1)
        v2 = v1 + self.b
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
