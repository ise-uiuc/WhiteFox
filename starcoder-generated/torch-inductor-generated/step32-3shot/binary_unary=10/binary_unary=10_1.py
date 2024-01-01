
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 8)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        if True:
            v2 = v1 + other
        else:
            v2 = v1 * other_2
        v3 = F.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 4)

# If condition is true
other = torch.randn(1, 8)
