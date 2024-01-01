
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 8, bias=True)
 
    def forward(self, x1):
        v1 = self.linear(x1).reshape(-1, 1, 2, 2)
        v2 = v1 - 1.0
        v3 = torch.relu(v2).squeeze()
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 16).reshape(-1, 4, 4)
