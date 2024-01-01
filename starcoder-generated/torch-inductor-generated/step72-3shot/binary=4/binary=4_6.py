
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 16, bias=False)
 
    def forward(self, x1, other1=None):
        if other1 is None:
            other1 = torch.ones(1)

        t1 = self.linear(x1)
        t2 = t1 + other1
        return t2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 16)
