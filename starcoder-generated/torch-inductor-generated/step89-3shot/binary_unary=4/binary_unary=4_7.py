
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(10, 10, False)
 
    def forward(self, input, other):
        v1 = self.linear(input)
        output = torch.nn.functional.relu(v1 + other)
        return output

# Initializing the model
m = Model()

# Inputs of the model
input = torch.randn(3, 10)
other = torch.randn(3, 10)
