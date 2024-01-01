
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 3)
 
    def forward(self, x1, x2, x3):
        x3 = 0.6 * x3
        x1 = 0.5 * x1
        l1 = self.linear(x1)
        l2 = self.linear(l1)
        v1 = torch.add(l2, x2, alpha=x3)
        v2 = torch.relu(v1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 1)
x2 = torch.randn(2, 3)
