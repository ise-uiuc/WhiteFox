
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        num_features = 120
        self.linear = torch.nn.Linear(num_features, 84)
        self.other  = torch.nn.Parameter(torch.zeros((84,), dtype=torch.float32))
        self.other.requires_grad_(True)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 - self.other
        v3 = F.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model 
x1 = torch.randn(1, 120)
