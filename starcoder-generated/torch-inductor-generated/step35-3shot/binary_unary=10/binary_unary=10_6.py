
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 2) # Input tensor will be initialized to a random value

    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + torch.ones_like(v1)
        v3 = F.relu(v2)
        return v3

# Initializing the model
m = Model()
n_input = 10

# Input to the model
x1 = torch.randn(1, n_input)
