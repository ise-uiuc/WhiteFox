
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
        self.other = other
 
    def forward(self, x2):
        v1 = self.linear(x2)
        v2 = v1 + self.other
        v3 = F.relu(v2)
        return v3

# Initializing the model
other = torch.tensor(1.0, requires_grad=True)
m = Model(other)

# Inputs to the model
x2 = torch.randn((1, 1))
