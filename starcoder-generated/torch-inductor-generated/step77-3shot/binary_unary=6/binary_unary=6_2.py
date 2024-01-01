
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.t1 = torch.nn.Linear(3, 4)
        self.other = other
 
    def forward(self, x1):
        v1 = self.t1(x1)
        v2 = v1 - self.other
        v3 = F.relu(v2)
        return v3

# Initializing the model
m = Model(1)

# Inputs to the model
x1 = torch.randn(1, 3)
