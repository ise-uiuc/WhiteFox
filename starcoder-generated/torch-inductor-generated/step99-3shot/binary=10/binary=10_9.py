
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(16, 5)
 
    def forward(self, x1):
        v1 = self.l1(x1)
        v2 = v1 + x1
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(5, 16)
other = torch.randn(5, 16)

__output = m(x1, other=other)

