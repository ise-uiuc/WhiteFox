
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 8)

    def forward(self, x1, other):
        v1 = self.linear(input)
        v2 = v1 - other
        v3 = relu(input)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64)
other = 3 
