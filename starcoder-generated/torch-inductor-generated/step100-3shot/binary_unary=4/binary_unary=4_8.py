
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
 
    def forward(self, x1, x2):
        o1 = self.linear(x1)
        o2 = o1 + x2
        o3 = o2.relu()
        return o3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 1, 3)
x2 = torch.randn(1, 1, 3)
