
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 20, bias=True)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v1 -= other
        return torch.relu(v1)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10)
