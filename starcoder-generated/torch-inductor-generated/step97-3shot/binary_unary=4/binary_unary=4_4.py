
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 10)
 
    def forward(self, input_tensor, other):
        x1 = self.linear(input_tensor)
        x2 = x1 + other
        x3 = torch.relu(x2)
        return x3

# Initializing the model
m = Model()

# Values to be passed as keyword arguments to forward (others)
other = torch.randn(10, 10)

# Inputs to the model
input_tensor = torch.randn(1, 3, 64, 64)
