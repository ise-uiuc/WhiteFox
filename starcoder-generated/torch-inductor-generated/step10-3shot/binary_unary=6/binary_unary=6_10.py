
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 8, bias=True)
 
    def forward(self, x2):
        v1 = self.linear(x2)
        other = torch.tensor([-0.2], dtype=torch.float32)
        v2 = v1 - other
        v3 = torch.sqrt(v2)
        v4 = torch.relu(v3)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(1, 16)
