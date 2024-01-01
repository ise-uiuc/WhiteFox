
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v4 = self.linear(x1, bias=other)
        v4 = self.linear(x1, bias=other)
        v2 = v1 + v4
        v3 = F.relu(v2)
        return v3

# Initializing the model
m = Model(other=torch.zeros(8))

# Inputs to the model
x1 = torch.randn(4, 3, 16, 64)
y1 = torch.randn(8)
