
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 6)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + other
        return v2

# Initializing the model
m = Model()

# Other
other = torch.Tensor(1, 6)

# Inputs to the model
x1 = torch.randn(1, 8)
