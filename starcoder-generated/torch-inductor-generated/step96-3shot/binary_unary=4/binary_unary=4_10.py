
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 32, bias=False)
 
    def forward(self, x1, other=None):
        if other is None:
            return self.linear(x1)
 
        v1 = self.linear(x1)
        v2 = v1 + other
        v3 = torch.nn.functional.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64)
other = torch.randn(1, 32)
