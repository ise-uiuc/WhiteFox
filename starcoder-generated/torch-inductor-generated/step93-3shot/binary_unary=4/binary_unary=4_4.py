
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)
 
    def forward(self, x1, other=torch.rand(4)):
        v1 = self.linear(x1)
        v2 = v1 + other
        v3 = F.relu(v2)
        return v3, v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 4)
__output__, output = m(x1)

