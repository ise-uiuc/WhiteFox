s
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 8)
 
    def forward(self, x):
        v1 = self.linear(x)
        return v1

class SecondModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 8)
 
    def forward(self, x, other):
        v1 = self.linear(x)
        v2 = v1 + other
        return v2

# Initializing the models
m = Model()
m_other_input = SecondModel()

# Inputs to the models
x = torch.randn(1, 16)
