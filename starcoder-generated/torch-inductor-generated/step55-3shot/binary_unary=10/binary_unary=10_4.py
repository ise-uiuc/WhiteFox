
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 2)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.randn_like(v1)
        v3 = self.linear(v2)
        v4 = v1 + v3
        v5 = torch.relu(v4)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10)
