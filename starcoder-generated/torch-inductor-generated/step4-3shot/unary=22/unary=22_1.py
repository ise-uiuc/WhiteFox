
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(30, 80)
 
    def forward(self, x0):
        a1 = self.linear(x0)
        a2 = torch.tanh(a1)
        return a2

# Initializing the model
m = Model()

# Inputs to the model
x0 = torch.randn(1, 30)
