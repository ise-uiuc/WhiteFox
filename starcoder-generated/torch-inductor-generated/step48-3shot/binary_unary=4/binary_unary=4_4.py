
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(64, 8)
        self.other = torch.nn.Parameter(other)
        # The following property is used for debugging purposes
        self.n_outputs = lambda x: (self.linear(x.view(-1,64))+self.other).shape[1]

    def forward(self, x1):
        v1 = self.linear(x1.view(-1,64))
        v2 = v1 + self.other
        v3 = torch.nn.functional.relu(v2)
        return v3

# Initializing the model
m = Model(torch.randn(8))

# Inputs to the model
x1 = torch.randn(5, 64)
