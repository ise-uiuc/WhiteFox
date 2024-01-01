
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8, bias=False)
 
    def forward(self, x1, other):
        v1 = self.linear(x1)
        v2 = v1 + other
        v3 = torch.nn.functional.relu(v2)
        return v3

# Initializing the model
other = torch.ones(1, 3, 8) # This value is ignored in the forward function
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
