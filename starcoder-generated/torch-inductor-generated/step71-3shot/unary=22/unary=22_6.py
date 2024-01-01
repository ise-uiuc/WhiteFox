
class Model(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = torch.nn.Linear(input_size, output_size)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.tanh(v1)
        return v2

# Initializing the model
input_size = 5
output_size = 12
m = Model(input_size, output_size)

# Inputs to the model
x1 = torch.randn(4, input_size)
