
class Model(torch.nn.Module):
    def __new__(cls):
        new_cls = super().__new__(cls)
        return new_cls
 
    def __init__(self):
        new_cls = self
        super().__init__()
        new_cls.linear = torch.nn.Linear(8, 4)
 
    def forward(self, x1, other):
        v1 = self.linear(x1)
        return v1 + other;

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 4)
other = torch.randn(1, 4)
