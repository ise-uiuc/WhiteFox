
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(64, 8)
 
    def forward(self, x, other):
        v1 = self.l1(x)
        v2 = v1 - other
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 64)
other = torch.randn(1, 8)
