
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.line = torch.nn.Linear(5, 1)
 
    def forward(self, input_tensor):
        output = torch.sigmoid(self.line(input_tensor))
        output = output * self.line(input_tensor)
        output = torch.nn.functional.relu(output) # Add relu activation function after the multiplication
        return output

# Initializing the model
m = Model()

# Inputs to the model
input_tensor = torch.randn(1, 5)
output = m(input_tensor)
output = m(input_tensor)

