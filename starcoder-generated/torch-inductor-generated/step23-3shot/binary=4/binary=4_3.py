
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 8)
        self.other = torch.ones(8)
 
    def forward(self, x0):
        v0 = self.linear(x0)
        v1 = v0 + self.other
        return v1

# Initializing the model
m = Model()

# Inputs to the model
x0 = torch.randn(2, 4)
