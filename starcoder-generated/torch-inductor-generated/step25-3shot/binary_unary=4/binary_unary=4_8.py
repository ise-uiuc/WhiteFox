
class Model(torch.nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
        self.weight = weight
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + self.weight
        return torch.relu(v2)

# Initializing the model
weight = torch.randn(8, 3)
m = Model(weight)

# Input to the model
x1 = torch.randn(8, 3)
