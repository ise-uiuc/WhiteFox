
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 1)
 
    def forward(self, x1):
        v6 = self.linear(x1)
        v7 = torch.tanh(v6)
        return v7

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 16)
