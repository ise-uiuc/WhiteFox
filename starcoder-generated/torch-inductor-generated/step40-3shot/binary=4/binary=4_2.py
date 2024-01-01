 (same with Model2)
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 10)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        return torch.relu(v1 + other)

# Initializing the model
m = Model()

# Inputs to the model (same with Model2)
x1 = torch.randn(1, 64, 1, 1)
