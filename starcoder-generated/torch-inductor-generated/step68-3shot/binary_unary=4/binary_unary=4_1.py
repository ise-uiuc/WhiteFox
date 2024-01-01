
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
        self.other = other
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + self.other
        v3 = v2.relu()
        return v3
 
# Initializing the model
m = Model(0.1)
 
# Inputs to the model
x1 = torch.randn(1, 1)
