
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 5)
 
    def forward(self, x1, other=None):
        if other is None:
            other = torch.tensor([1, 2, 3, 4, 5])
        v1 = self.linear(x1)
        v2 = v1 + other
        v3 = F.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.tensor([0.1])
