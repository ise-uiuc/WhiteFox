
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 5)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + torch.tensor([1, 2, 3, 4, 5], dtype=torch.float)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 5)
a = torch.tensor([10, 100, 1000, 10000, 100000], dtype=torch.float)
