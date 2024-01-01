
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 1)
 
    def forward(self, input_tensor, other):
        output = self.linear(input_tensor)
        output = output - other
        return output

# Initializing the model
m = Model()

# Inputs to the model
input_tensor = torch.randn(1, 3)
other = 1.0  # a floating point value
