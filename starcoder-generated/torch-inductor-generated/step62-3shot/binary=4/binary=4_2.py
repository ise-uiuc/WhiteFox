
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(784, 10)
 
    def forward(self, input_tensor, other):
        output = self.linear(input_tensor)
        output += other
        return output

# Initializing the model
m = Model()

# Inputs to the model
input_tensor = torch.randn(32, 784)
other = torch.randn(32, 10)
output = m(input_tensor, other)

