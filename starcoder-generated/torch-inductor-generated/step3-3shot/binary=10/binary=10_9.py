
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Parameter(torch.rand(32, 11))

    def forward(self, inputs):
        return self.linear + inputs

# Initialize the model
m = Model()

# Inputs to the model
inputs = torch.randn(1, 11, 32,)
out = m(inputs) # m(input) == m(input[None])[0]

