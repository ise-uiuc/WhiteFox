
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 32)
 
    def forward(self, x1):
        x2 = self.linear(x1)
        x3 = x2 * 0.5
        x4 = x2 + (x2 * x2 * x2) * 0.044715
        x5 = x4 * 0.7978845608028654
        x6 = torch.tanh(x5)
        x7 = x6 + 1
        x8 = x3 * x7
        return x8

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 16)
