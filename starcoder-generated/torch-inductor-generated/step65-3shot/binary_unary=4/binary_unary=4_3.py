
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(4, 5)
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = v1 + other
        v3 = torch.relu(v2)
        return v3
 
# Initializing the model
m = Model(x)

# Inputs to the model
x = torch.randn(1, 4)
