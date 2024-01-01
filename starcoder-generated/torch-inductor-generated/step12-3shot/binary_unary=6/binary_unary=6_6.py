
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l = torch.nn.Linear(3, 8)
 
    def forward(self, x1, other):
        v1 = self.l(x1)
        v2 = v1 - other
        v3 = torch.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
other = torch.tensor([1.1, 0.2, 3.3])
