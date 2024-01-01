
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        return torch.relu(v1 + other)

# Initializing the model and input tensor(s)
m = Model(other=torch.randn(1, 8))

# Inputs to the model
x1 = torch.randn(1, 3)
