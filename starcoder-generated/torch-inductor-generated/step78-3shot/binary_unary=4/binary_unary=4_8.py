
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
 
    def forward(self, input, other=None):
        x = self.linear(input)
        if other is not None:
            x = x + other
        x = torch.relu(x)
        return x

# Initializing the model with a random tensor for the keyword argument `other`
m = Model()
other = torch.rand(8)

# Inputs to the model
input = torch.randn(3)
