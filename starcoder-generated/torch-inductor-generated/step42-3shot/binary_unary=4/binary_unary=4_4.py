
class Model(torch.nn.Module):
    def __init__(self, linear = torch.nn.Linear(5551, 5551)):
        super().__init__()
        self.linear = linear
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + other
        v3 = F.relu(v2)
        return v3

# Initializing the model
m = Model(torch.nn.Linear(5551, 5551))

# Inputs to the model with default weights
x1 = torch.randn(1, 11715)
