
class Model(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
 
    def forward(self, x1):
        return torch.relu(self.linear(x1))

# Initializing the model
m = Model(10, 4)

# Inputs to the model
x1 = torch.randn(1, 10)
