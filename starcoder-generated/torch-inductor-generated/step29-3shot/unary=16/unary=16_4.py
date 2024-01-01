
class Model(torch.nn.Module):
    def __init__(self, dim, input_size, output_size):
        super().__init__()
        self.linear = torch.nn.Linear(input_size, output_size)
 
    def forward(self, x2):
        v7 = self.linear(x2)
        v8 = torch.nn.functional.relu(v7)
        return v8

# Initializing the model
m = Model(dim=128, input_size=8, output_size=64)

# Inputs to the model
x2 = torch.randn(10, 8)
