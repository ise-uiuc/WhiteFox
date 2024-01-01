
class Model(torch.nn.Module):
    def __init__(self, other=None):
        super().__init__()
        self.linear1 = torch.nn.Linear(3, 8)
 
    def forward(self, x1):
        v1 = self.linear1(x1)
        v2 = v1 + other
        v3 = torch.relu(v2)
        return v3

# Initializing the model
other = torch.randn(8, 8)
m = Model(other=other)

# Inputs to the model
x1 = torch.randn(1, 3, 4, 4)
