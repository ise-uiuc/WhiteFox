
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
        self.other = torch.zeros(8)
 
    def forward(self, __input__):
        o1 = self.linear(__input__)
        o2 = o1 + self.other
        return o2

# Initializing the model
m = Model()

# Inputs to the model
__input__ = torch.randn(1, 3)

# Output of model
