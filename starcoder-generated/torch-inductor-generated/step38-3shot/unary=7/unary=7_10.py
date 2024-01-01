
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 3)
 
    def forward(self, x1):
        o1 = self.linear(x1)
        o2 = torch.clamp(o1 + 3, min=0, max=6) * o1
        return o2 / 6

# Initializing the model
m = Model()

# Generating the input to the model
x1 = torch.randn(2, 10)

# Get the output of the model
