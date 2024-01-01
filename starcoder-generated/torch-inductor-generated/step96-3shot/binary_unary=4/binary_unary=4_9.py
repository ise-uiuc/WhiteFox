
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(7, 7)
        self.other = other
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + self.other
        v3 = F.relu(v2)
        return v3

# Initializing the model
f1 = torch.randn(1, 7)
m1 = Model(other=f1)

# Inputs to the model
x1 = torch.randn(1, 7)
