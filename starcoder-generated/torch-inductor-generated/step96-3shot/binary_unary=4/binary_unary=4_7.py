
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 16)
    
    def forward(self, x1, x2=torch.randn(1, 16)):
        o = self.linear(x1)
        o += x2
        o = torch.nn.functional.relu(o)
        return o

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 16)
x2 = torch.randn(1, 16)
