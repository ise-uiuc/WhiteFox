
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(256, 8)
 
    def forward(self, x1, other):
        v3 = self.linear(x1)
        v1 = v3 + other
        return v1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 256)
other = torch.randn(1, 8)
