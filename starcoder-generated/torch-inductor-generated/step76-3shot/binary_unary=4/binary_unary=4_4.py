
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 32)
 
    def forward(self, x1, other=None):
        t1 = self.linear(x1)
        t2 = t1 + other
        return F.relu(t2)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8)
