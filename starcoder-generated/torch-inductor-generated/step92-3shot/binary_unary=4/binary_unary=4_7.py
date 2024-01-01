
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(2, 4)
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = v1 + other
        v3 = v2 * relu(other)
        return v3

# Initializing the model and defining the value of the other tensor
m = Model(torch.tensor(1.0))

# Inputs to the model
x = torch.randn(2)

