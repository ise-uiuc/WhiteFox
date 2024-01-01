
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)
        self.other = other
 
    def forward(self, input_tensor):
        linear_output = self.linear(input_tensor)
        out = linear_output - self.other
        return out

# Initializing the model
m = Model(torch.randn(10))

# Input to the model
input_tensor = torch.randn(1, 10)

