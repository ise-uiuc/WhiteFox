
class Model(torch.nn.Module):
    def __init__(self, other_value):
        super().__init__()
        self.linear = torch.nn.Linear(6, 8)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 - self.other_value
        v3 = torch.relu(v2)
        return v3

# Initializing the model
m = Model(other_value=1)

# Inputs to the model
x1 = torch.randn(1, 6)
