
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 16)
 
    def forward(self, x1):
        x2 = self.linear(x1)
        x3 = torch.relu(x2)
        return x1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1000, 16)
