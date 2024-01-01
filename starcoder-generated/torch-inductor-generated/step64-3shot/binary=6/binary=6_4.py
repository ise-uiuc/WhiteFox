
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
 
    def forward(self, x1, p2):
        o1 = self.linear(x1)
        o2 = o1 - p2
        return o2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
p2 = torch.tensor([-0.1, 0.2, 2.0])
