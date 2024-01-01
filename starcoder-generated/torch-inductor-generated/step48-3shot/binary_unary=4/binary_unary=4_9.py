
class LinearPlusReluModel(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(16, 8)
        self.other = other
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = v1 + self.other
        v3 = torch.relu(v2)
        return v3
 
# Initializing the model
m = LinearPlusReluModel(torch.randn(8))

# Inputs to the model
x1 = torch.randn(1, 8, 64, 64)
