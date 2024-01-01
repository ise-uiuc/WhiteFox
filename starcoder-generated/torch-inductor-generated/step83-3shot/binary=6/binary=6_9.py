
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 - 1
        return torch.relu(v2)

# Initializing the model
m = Model()

# Input to the model
x1 = torch.randn(5, 10)
