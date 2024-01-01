
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)
 
    def forward(self, x1):
        t1 = self.linear(x1)
        t2 = t1 * 0.5
        t3 = t1 * 0.7071067811865476
        t4 = torch.erf(t3)
        t5 = t4 + 1
        t6 = t2 * t5
        return t6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 10)
