
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 64)
 
    def forward(self, x2):
        v5 = self.linear(x2)
        v6 = torch.tanh(v5)
        return v6

# Initializing the model
m1 = Model()

# Inputs to the model
x2 = torch.randn(3, 16)
