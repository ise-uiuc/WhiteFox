
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 30)
 
    def forward(self, x1, x2):
        v1 = self.linear(x1)
        v2 = v1 + x2
        return torch.relu(v2)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(784)
x2 = torch.randn(30)
