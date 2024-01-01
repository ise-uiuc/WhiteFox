
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 5)
 
    def forward(self, x2):
        v2 = self.linear(x2)
        return v2 + x2

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(2, 3)
other = torch.ones(3)
