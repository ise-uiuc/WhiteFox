
class GatedTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 8)
 
    def forward(self, x):
        x1 = self.linear(x)
        x2 = torch.sigmoid(x1)
        return x1 * x2

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
        self.gate = GatedTanh()
 
    def forward(self, x):
        x1 = self.linear(x)
        x2 = self.gate(x1)
        return x2

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 1)
