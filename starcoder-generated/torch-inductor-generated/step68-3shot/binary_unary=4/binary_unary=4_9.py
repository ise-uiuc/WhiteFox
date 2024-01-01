
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(5, 2048)
 
    def forward(self, x1, x2):
        v1 = self.linear(x1)
        v2 = v1 + x2
        v3 = torch.relu(v2)
        return v3

# Initializing the model
m = Model(torch.randint(-2, 2, (1, 5)))

# Inputs to the model
x1 = torch.randint(0, 1, (1, 5))
x2 = torch.randint(-2, 2, (1, 2048))
