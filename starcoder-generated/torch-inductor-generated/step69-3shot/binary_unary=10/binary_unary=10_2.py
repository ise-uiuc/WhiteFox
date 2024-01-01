
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(64, 32)
        
    def forward(self, x1):
        v1 = self.linear1(x1)
        v2 = v1 + torch.ones(1, 32)
        v3 = torch.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(10, 64)
