
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
        self.other = other
 
    def forward(self, inputs):
        output = self.linear(inputs)
        output += self.other
        return output

# Initializing the model
m = Model(torch.randn(1, 3, 64, 64))

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
