
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(15, 10)
 
    def forward(self, x1, _other):
        v1 = self.linear(x1)
        v2 = v1 + _other
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 15)
other = torch.randn(1, 10)
