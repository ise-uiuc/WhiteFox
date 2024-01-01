
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        f = 32
        self.linear = torch.nn.Linear(32,f)
        self.linear_other = torch.nn.Linear(f, 2)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + self.linear_other
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 32)
