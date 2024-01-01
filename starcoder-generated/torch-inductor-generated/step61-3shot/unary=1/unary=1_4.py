
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 16)
 
    def forward(self, x1):
        q = self.linear(x1)
        h1 = q * 0.5
        h2 = (q * q * q) * 0.044715
        h3 = h2 + h1
        h4 = h3 * 0.7978845608028654
        e = torch.tanh(h4)
        y = e + 1
        v = q * y
        return v

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 1)
