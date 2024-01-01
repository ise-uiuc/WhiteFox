
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
 
    def forward(self, _input):
        v1 = self.linear(_input)
        v2 = v1 - other
        return v2

# Initializing the model
m = Model()

# Inputs to the model
__input01__ = torch.randn(1, 3)
__input02__ = torch.randn(3)
v1 = __input02__
