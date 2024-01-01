
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
 
    def forward(self, x1, other):
        v7 = other.view(8)
        v1 = self.linear(x1)
        v2 = v1 - v7
        v3 = torch.relu(v2)
        return v3

# Initializing the model
other = torch.randn(8, )
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
