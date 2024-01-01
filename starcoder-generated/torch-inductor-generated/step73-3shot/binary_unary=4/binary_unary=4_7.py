
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(32, 64)
        self.other = other
 
    def forward(self, x):
        y = self.linear(x)
        z = y + self.other
        return torch.relu(z)
 
# Initializing the model
m = Model(other=torch.randn(64, 64))
 
# Inputs to the model
x = torch.randn(10, 32)
