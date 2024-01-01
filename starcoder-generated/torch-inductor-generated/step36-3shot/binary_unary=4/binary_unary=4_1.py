
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 3)
 
    def forward(self, x1, other: torch.Tensor):
        v1 = self.linear(x1)
        v2 = v1 + other
        v3 = torch.relu(v2)
        return v3
 
# Initializing the model
m = Model()

# Inputs to the model
other = torch.rand(3, 2)
x1 = torch.rand(2, 2)
