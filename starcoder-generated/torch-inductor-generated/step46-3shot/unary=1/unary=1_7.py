
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
 
    def forward(self, x1):
        t1 = self.linear(x1)
        t2 = t1 * 0.5
        t3 = t1 + (t1 * t1 * t1) * 0.044715
        t4 = t3 * 0.7978845608028654
        t5 = torch.tanh(t4)
        t6 = t5 + 1
        t7 = t2 * t6
        return t7

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
