
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(100, 10)
 
    def forward(self, x):
        z = self.linear(x)
        c = torch.cos(z * z)
        t1 = z * 0.5
        t2 = z + (z * z * z) * 0.044715
        t3 = t2 * 0.7978845608028654
        t4 = torch.tanh(t3)
        t5 = t4 + 1
        t6 = t1 * t5
        return t6

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(100, 10)
