
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
 
    def forward(self, x1):
        t1 = self.linear(x1)
        t2 = t1 - 0.01
        t3 = torch.relu(t2)
        return t3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 1)
