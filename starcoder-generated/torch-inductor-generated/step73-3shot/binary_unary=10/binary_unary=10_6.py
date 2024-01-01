
class Model(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.linear = torch.nn.Linear(d, 5)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        d = 4
        v2 = v1 + torch.tensor(d)
        v3 = torch.relu(v2)
        return v3

# Initializing the model
d = 3
m = Model(d)

# Inputs to the model
x1 = torch.randn(1, d)
