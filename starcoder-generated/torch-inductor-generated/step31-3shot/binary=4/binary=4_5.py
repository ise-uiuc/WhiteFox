
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(8, 8)
 
    def forward(self, x1, x2):
        i1 = self.l1(x1)
        o = i1 + x2
        return o

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8)
x2 = torch.randn(1, 8)
