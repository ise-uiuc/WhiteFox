
class Model(torch.nn.Module):
    def __init__(self, other=3.141592653589793):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)
        self.linear.weight.data.fill_(3.141592653589793)
        self.linear.bias.data.fill_(10)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 - self.other # The 'other' value is specified as a module attribute
        v3 = F.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 2)
