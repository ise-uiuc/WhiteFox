
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        m, p = 6, 7
        self.linear = torch.nn.Linear(m, p, bias=False)
 
    def forward(self, x): 
        v1 = self.linear(x)
        v2 = v1 + other
        return v2

# Initializing the model
m = Model()

# Inputs to the model
other = torch.randn(p, m)
x = torch.randn(1, m)
