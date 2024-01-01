
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
 
    def forward(self, x1):
        t1 = self.linear(x1)
        t2 = t1 - 0.2
        t3 = torch.relu(t2)
        v1 = t3 + 1 # add one
        return v1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(10, 3)
