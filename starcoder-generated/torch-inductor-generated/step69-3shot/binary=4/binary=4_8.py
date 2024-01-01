
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 2)
 
    def forward(self, x1, x2):
        add = torch.add(x1, x2)
        v1 = self.linear(add)
        return v1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8)
x2 = torch.randn(1, 8)
