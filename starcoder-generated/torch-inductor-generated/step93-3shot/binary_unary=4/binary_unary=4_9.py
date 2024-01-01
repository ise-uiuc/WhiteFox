
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(10, 10)
 
    def forward(self, input, other):
        v1 = self.linear(input)
        v2 = v1 + other
        v3 = torch.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
input = torch.randn(10)
other = torch.randn(10)
