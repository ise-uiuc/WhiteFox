
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear()
 
    def forward(self, x1, other=None):
        v1 = self.linear(x1)
        v2 = v1 + other
        v3 = torch.relu(v2)
        return v3

# Initializing the model
m = Model(other=torch.randn(5))

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
