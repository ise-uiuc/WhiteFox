
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)
 
    def forward(self, input, other):
        y = self.linear(input)
        z = y + other
        a = F.relu(z)
        return a

# Initializing the model
m = Model()

# Inputs to the model
input = torch.randn(16, 10)
other = torch.randn(16, 10)
