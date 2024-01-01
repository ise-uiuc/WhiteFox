
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(32, 32)
 
    def forward(self, x):
        v = self.linear(x)
        v = torch.tanh(v)
        return v

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 32)
