
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
 
    def forward(self, x1):
        kwargs = {'other': x1}
        v1 = self.linear(x1, **kwargs)
        v2 = v1 + x1
        v3 = torch.relu(v2)
        return v3

# Initializing the model
other = torch.randn(1, 8)
m = Model(other)

# Inputs to the model
x1 = torch.randn(1, 3)
