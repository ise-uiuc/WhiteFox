
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)
 
    def forward(self, x1):
        y = self.linear(x1)
        t1 = y * 0.5
        t2 = y + (y * y * y) * 0.044715
        t3 = t2 * 0.7978845608028654
        t4 = torch.tanh(t3)
        t5 = t4 + 1
        t6 = t1 * t5
        return t6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10)
