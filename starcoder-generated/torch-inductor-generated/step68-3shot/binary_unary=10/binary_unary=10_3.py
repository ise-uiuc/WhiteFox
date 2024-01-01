
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
 
    def forward(self, x1: torch.Tensor):
        v1 = self.linear(x1)
        v2 = v1 + x1[:, 0:8]
        return torch.relu(v2)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
