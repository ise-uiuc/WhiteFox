
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 7)
 
    def forward(self, __input0, other):
        v1 = self.linear(__input0)
        v2 = v1 + other
        return v2

# Initializing the model
m = Model()

# Inputs to the model
input0 = torch.randn(1, 5, 8, 8)
other = torch.randn(1, 7, 4, 4)
