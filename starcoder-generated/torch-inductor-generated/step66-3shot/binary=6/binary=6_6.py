
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 3)
 
    def forward(self, x1):
        s1 = self.linear(x1)
        s2 = s1 - torch.tensor(1.)
        return s2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 2)
