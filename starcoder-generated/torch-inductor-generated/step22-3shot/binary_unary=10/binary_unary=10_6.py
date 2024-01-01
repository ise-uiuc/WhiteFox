s

## Model A
class ModelA(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1000000, 1000)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + torch.full_like(v1, 1000000,  dtype=torch.float)
        v3 = F.relu(v2)
        return v3

# Initializing the model
m = ModelA()

# Inputs to the model
x1 = torch.randn(1, 1000000,  dtype=torch.float)
