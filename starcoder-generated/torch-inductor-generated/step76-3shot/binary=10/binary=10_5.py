
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 16)
 
    def forward(self, x, weight):
        v1 = self.linear(x)
        v2 = v1 + weight
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(3, 3, 3, 3)
weight = torch.tensor([1, 1, 1, 1])
