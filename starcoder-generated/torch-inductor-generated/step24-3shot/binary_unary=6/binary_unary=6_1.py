
class Model(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = torch.nn.Linear(input_size, output_size)
 
    def forward(self, x, other):
        v1 = self.linear(x)
        v2 = v1 - other
        v3 = torch.relu(v2)
        return v3


# Initializing the model
m = Model(16, 16)

# Inputs to the model
x = torch.randn(1, 16)
other = torch.randn(1, 16)
