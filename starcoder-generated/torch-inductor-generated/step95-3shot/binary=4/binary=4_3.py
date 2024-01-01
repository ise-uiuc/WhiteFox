
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)
 
    def forward(self, linear_input, add_tensor, other):
        v1 = self.linear(linear_input)
        v2 = v1 + add_tensor
        v3 = v2 + other
        return v3

# Initializing the model
m = Model()

# Input to the model
linear_input = torch.randn(1, 10)
# One way to pass the model's input
