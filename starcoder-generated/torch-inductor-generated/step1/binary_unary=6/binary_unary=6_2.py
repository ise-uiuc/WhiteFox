
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(128, 128)
 
    def forward(self, x, other):
        return torch.relu(self.linear(x) - other)

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 128, 28, 28)
other = torch.randn(1, 128)
