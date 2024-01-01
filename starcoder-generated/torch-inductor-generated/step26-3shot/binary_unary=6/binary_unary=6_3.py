
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = Linear(2, 4)

    def forward(self, x):
        v1 = self.linear(x)
        v2 = v1 - other
        v3 = torch.nn.functional.relu(v1)
        return v3

# Initializing the model and other
m = Model()

other = __import__('random').uniform() # the randomly generated value used in the pattern

# Inputs to the model
x = torch.randn(1, 2)
