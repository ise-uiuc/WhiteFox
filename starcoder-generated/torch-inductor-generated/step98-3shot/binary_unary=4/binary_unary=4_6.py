
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.linear(1000, 4096)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1
        v3 = v2 + other
        v4 = F.relu(v3)
        return v4

# Initializing the model
m = Model(other=torch.rand(1, 4096))

# Inputs to the model
x1 = torch.randn(1, 1000)
