
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.other = other
 
    def forward(self, x1):
        v1 = F.linear(x1, self.other)
        v2 = v1 + self.other
        v3 = F.relu(v2)
        return v3

# Initializing the model
other = torch.randn(1, 4)
m = Model(other)

# Inputs to the model
x1 = torch.randn(1, 3)
