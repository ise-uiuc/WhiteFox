
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 10)
 
    def forward(self, input, other=None):
        v1 = self.linear(input)
        if other is not None:
            v1 = v1 + other
        v2 = F.relu(v1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(10, 4)
x2 = torch.randn(10, 10)
