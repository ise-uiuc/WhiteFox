
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(3, 5)

    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = {'other' : x1, 'v3' : v1} # add a new entry to the output of the linear tensor
        v3 = torch.relu(v2['v3'])
        return v3

# Initializing the model
m = Model(other)

# Inputs to the model
x1 = torch.randn(1, 3)
