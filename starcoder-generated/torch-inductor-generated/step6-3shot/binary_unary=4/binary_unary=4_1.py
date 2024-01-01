
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(5, 1)
 
    def forward(self, x1, other):
        v1 = self.linear(x1)
        v2 = v1 + other
        v3 = F.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.rand(1, 5)
other = torch.rand(1, 1)
