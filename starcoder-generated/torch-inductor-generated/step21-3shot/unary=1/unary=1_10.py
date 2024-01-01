
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 512)
 
    def forward(self, x1):
        a1 = self.linear(x1)
        a2 = a1 * 0.5
        a3 = a1 + (a1 * a1 * a1) * 0.044715
        a4 = a3 * 0.7978845608028654
        a5 = torch.tanh(a4)
        a6 = a5 + 1
        a7 = a2 * a6
        return a7

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
