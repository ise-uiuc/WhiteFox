
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(16, 16, bias=False)
        self.other = other
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + self.other
        v3 = F.relu(v2)
        return v3

# Initializing the model
m = Model(torch.randn(16))

# Inputs to the model
x1 = torch.randn(100, 16)
