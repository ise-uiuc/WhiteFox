
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(32, 32)
 
    def forward(self, x, other):
        v1 = self.linear(x)
        v2 = v1 + other
        return F.relu(v2)

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(2, 32)
other = torch.randn(1, 32)
