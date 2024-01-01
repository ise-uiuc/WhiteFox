
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.l1 = torch.nn.Linear(8, 8)
        self.other = other
 
    def forward(self, x1):
        x1 = self.l1(x1)
        x1 = x1 + self.other
        x1 = torch.nn.functional.relu(x1)
        return x1

# Initializing the model
m = Model(torch.randn(8, 8))

# Inputs to the model
x1 = torch.randn(1, 8)
