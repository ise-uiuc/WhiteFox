
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.fc1 = torch.nn.Linear(12, 3)
        self.other = other
 
    def forward(self, x1):
        x2 = self.fc1(x1)
        v1 = x2 + self.other
        return v1

# Initializing the model
m = Model(other)

# Inputs to the model
x1 = torch.randn(2, 12)
