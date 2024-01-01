
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
 
    def forward(self, x1, other=None):
        x2 = self.linear(x1)
        x3 = x2 + other
        x4 = F.relu(x3)
        return x4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
