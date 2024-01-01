
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear_layer = torch.nn.Linear(369, 5)
        self.other = other
 
    def forward(self, x1):
        v1 = self.linear_layer(x1)
        v2 = v1.add(self.other)
        v3 = torch.relu(v2)
        return v3

# Initializing the model
m = Model(torch.tensor([[0.1, 0.5, 1.2]]))

# Inputs to the model
x1 = torch.randn(1, 369)
